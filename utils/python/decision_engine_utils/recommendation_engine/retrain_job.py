import argparse
import logging
import sys
import hopsworks
import yaml
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, col, expr
import tensorflow as tf
import tensorflow_addons as tfa

from hsml.client.exceptions import ModelServingException

logging.basicConfig(level=logging.INFO)
spark = SparkSession.builder.enableHiveSupport().getOrCreate()
BATCH_SIZE = 2048


def df_to_ds(df):
    return tf.data.Dataset.from_tensor_slices({col: df[col] for col in df})


def login_to_project(args):
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    ms = project.get_model_serving()
    dataset_api = project.get_dataset_api()
    downloaded_file_path = dataset_api.download(
        f"Resources/decision-engine/{args.name}/configuration.yml"
    )
    with open(downloaded_file_path, "r") as f:
        config = yaml.safe_load(f)
    prefix = "de_" + config["name"] + "_"
    return project, fs, mr, ms, prefix, config


def load_data(fs, prefix, config):
    events_fg = fs.get_feature_group(name=prefix + "events")
    events_df = events_fg.read()

    items_fg = fs.get_feature_group(
        name=prefix + config["product_list"]["feature_view_name"]
    )
    items_df = items_fg.read()
    return events_fg, events_df, items_fg, items_df


def compute_fg_size(events_fg):
    try:
        fg_stat = events_fg.statistics
    except Exception as e:
        logging.info(f"Couldn't retrieve events FG statistics. Quitting. Error: {e}")
        # no data available yet - no events received
        sys.exit()

    fg_size = next(
        column["count"]
        for column in fg_stat.to_dict()["featureDescriptiveStatistics"]
        if column["featureName"] == "request_id"
    )
    logging.info(f"Events feature group size is {fg_size} rows")
    return fg_size


def compute_td_size(fs, fg_name):
    events_fv = fs.get_feature_view(name=fg_name)
    td_list = events_fv.get_training_datasets()

    td_version = max(events_fv.get_training_datasets(), key=lambda m: m.version).version
    logging.info(f"Retrieving training dataset of version {td_version}")

    ds_size = 0  # training dataset size
    train_test_stats = [
        m.feature_descriptive_statistics
        for m in events_fv.get_training_dataset_statistics(td_version).split_statistics
    ]
    for feat_list in train_test_stats:
        for feature in feat_list:
            if feature.to_dict()["featureName"] == "request_id":
                ds_size += feature.to_dict()["count"]
    logging.info(f"Training dataset size is {ds_size} rows")
    return ds_size


def preprocess_retrieval_train_df(events_df, items_df):
    session_counts = events_df.groupBy("session_id").count()
    multi_interaction_sessions = session_counts.filter(
        session_counts["count"] > 5
    )  # min session length
    filtered_df = events_df.join(multi_interaction_sessions, "session_id", "inner")
    aggregated_df = filtered_df.groupby("session_id").agg(
        collect_set(col("item_id")).alias("item_ids")
    )
    retrieval_train_df = aggregated_df.join(
        items_df,
        col("item_ids").contains(items_df[config["product_list"]["primary_key"]]),
        "inner",
    )
    train_df, val_df = retrieval_train_df.randomSplit([0.8, 0.2], seed=123)

    train_ds = df_to_ds(train_df).batch(BATCH_SIZE).cache().shuffle(BATCH_SIZE * 10)
    val_ds = df_to_ds(val_df).batch(BATCH_SIZE).cache()
    return train_ds, val_ds


def train_model(train_ds, val_ds, mr, model_name):
    mr_model = mr.get_model(name=model_name, version=1)
    path = mr_model.download()
    new_model = tf.saved_model.load(path)
    optimizer = tfa.optimizers.AdamW(0.001, learning_rate=0.01)
    new_model.compile(optimizer=optimizer)
    new_model.fit(train_ds, validation_data=val_ds, epochs=5)
    tf.saved_model.save(new_model, "retrained_model")
    return new_model


def preprocess_ranking_train_df(events_df, items_df, config):
    if "policy_config" in config["model_configuration"]["ranking_model"].keys():
        events_df_pivoted = (
            events_df.groupBy("session_id", "item_id")
            .pivot("event_type")
            .agg(sum("event_value"))
            .fillna(0)
        )  # fill NaN values with 0
        # TODO how to agg session features
        items_scores = events_df_pivoted.withColumn(
            "score",
            expr(config["model_configuration"]["ranking_model"]["policy_config"]),
        )
    else:
        items_scores = events_df.groupBy("session_id", "item_id").agg(
            sum("event_weight").alias("score")
        )

    ranking_train_df = items_scores.join(
        events_df,
        items_df[config["product_list"]["primary_key"]] == events_df["item_id"],
        "inner",
    )
    train_df, val_df = ranking_train_df.randomSplit([0.8, 0.2], seed=123)

    train_ds = df_to_ds(train_df).batch(BATCH_SIZE).cache().shuffle(BATCH_SIZE * 10)
    val_ds = df_to_ds(val_df).batch(BATCH_SIZE).cache()
    return train_ds, val_ds


def save_model_to_registry(model, mr, model_name):
    retrained_model = mr.tensorflow.create_model(
        name=model_name,
        description=model.description,
        input_example=model.input_example,
        model_schema=model.model_schema,
    )
    retrained_model.save("retrained_model")
    logging.info(f"New {model_name} model version is {retrained_model.version}")
    return retrained_model.version


def update_deployment(ms, project, new_version, model_name, deployment_name):
    deployment = ms.get_deployment()
    deployment.model_version = new_version
    deployment.artifact_version = "CREATE"
    deployment.script_file = os.path.join(
        "/Projects", project.name, "Resources", f"{model_name}_predictor.py"
    ).replace("\\", "/")

    logging.info(f"New deployment properties: {deployment.to_dict()}")

    try:
        deployment.save(await_update=120)
    except ModelServingException as e:
        logging.info(f"deployment.save(await_update=120) failed. {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Name of DE project", default="none")
    args = parser.parse_args()

    project, fs, mr, ms, prefix, config = login_to_project(args)
    events_fg, events_df, items_fg, items_df = load_data(fs, prefix, config)

    if config["model_configuration"]["retrain"]["type"] == "ratio_threshold":
        fg_size = compute_fg_size(events_fg)
        td_size = compute_td_size(fs, prefix + "events")

        # Comparing data
        size_ratio = fg_size / td_size
        if (
            not size_ratio
            or size_ratio < config["model_configuration"]["retrain"]["parameter"]
        ):
            logging.info(
                "The new data is less than ratio threshold of the current dataset size. Quitting."
            )
            sys.exit()

    # Retrieval model
    train_ds, val_ds = preprocess_retrieval_train_df(events_df, items_df)
    retrieval_model = train_model(train_ds, val_ds, mr, prefix + "retrieval_model")
    new_version = save_model_to_registry(retrieval_model, mr, prefix + "retrieval_model")
    update_deployment(
        ms,
        project,
        new_version,
        prefix + "retrieval_model",
        (prefix + "retrieval_model").replace("_", "").lower(),
    )
    # TODO Opensearch update index

    # Ranking model
    train_ds, val_ds = preprocess_ranking_train_df(events_df, items_df, config)
    ranking_model = train_model(train_ds, val_ds, mr, prefix + "ranking_model")
    new_version = save_model_to_registry(ranking_model, mr, prefix + "ranking_model")
    update_deployment(
        ms,
        project,
        new_version,
        prefix + "ranking_model",
        (prefix + "ranking_deployment").replace("_", "").lower(),
    )
