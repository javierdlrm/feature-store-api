import argparse
import json
import logging
import sys
import hopsworks
import yaml
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, col, expr
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_recommenders as tfrs

from opensearchpy import OpenSearch

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
    opensearch_api = project.get_opensearch_api()
    dataset_api = project.get_dataset_api()
    downloaded_file_path = dataset_api.download(
        f"Resources/decision-engine/{args.name}/configuration.yml"
    )
    with open(downloaded_file_path, "r") as f:
        config = yaml.safe_load(f)
    prefix = "de_" + config["name"] + "_"
    return project, fs, mr, ms, prefix, config, opensearch_api


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


def preprocess_retrieval_train_df(events_df, items_df, config):
    session_counts = events_df.groupBy("session_id").count()
    multi_interaction_sessions = session_counts.filter(
        session_counts["count"] > 5
    )  # min session length
    filtered_df = events_df.join(multi_interaction_sessions, "session_id", "inner")
    aggregated_df = filtered_df.groupby("session_id").agg(
        collect_set(col("item_id")).alias("context_item_ids")
    )
    retrieval_train_df = aggregated_df.join(
        items_df,
        col("context_item_ids").contains(items_df[config["product_list"]["primary_key"]]),
        "inner",
    )
    train_df, val_df = retrieval_train_df.randomSplit([0.8, 0.2], seed=123)
    train_ds = df_to_ds(train_df).batch(BATCH_SIZE).cache().shuffle(BATCH_SIZE * 10)
    val_ds = df_to_ds(val_df).batch(BATCH_SIZE).cache()
    # TODO timestamp features needs transformation
    return train_ds, val_ds


class RetrievalModel(tfrs.Model):
    """
    Two-tower Retrieval model.
    """

    def __init__(self, query_model, candidate_model, catalog_ds):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=catalog_ds.batch(128).map(self._candidate_model)
            )
        )

    def compute_loss(self, features, training=False):
        context_item_ids = features.pop("context_item_ids")
        label_item_features = features

        query_embedding = self._query_model(context_item_ids)
        candidate_embedding = self._candidate_model(label_item_features)

        return self._task(
            query_embedding, candidate_embedding, compute_metrics=not training
        )


def train_retrieval_model(catalog_ds, train_ds, val_ds, mr, prefix):
    mr_query_model = mr.get_model(name=prefix + "query_model", version=1)
    path = mr_query_model.download()
    new_query_model = tf.saved_model.load(path)
    
    mr_candidate_model = mr.get_model(name=prefix + "candidate_model", version=1)
    path = mr_candidate_model.download()
    new_candidate_model = tf.saved_model.load(path)
    
    retrieval_model = RetrievalModel(new_query_model, new_candidate_model, catalog_ds) 
    optimizer = tfa.optimizers.AdamW(0.001, learning_rate=0.01)
    retrieval_model.compile(optimizer=optimizer)
    retrieval_model.fit(train_ds, validation_data=val_ds, epochs=5)
    
    tf.saved_model.save(new_query_model, prefix + "query_model")
    tf.saved_model.save(new_candidate_model, prefix + "candidate_model")
    return new_query_model, new_candidate_model


def train_ranking_model(train_ds, val_ds, mr, prefix):
    mr_model = mr.get_model(name=prefix + "ranking_model", version=1)
    path = mr_model.download()
    new_model = tf.saved_model.load(path)
    optimizer = tfa.optimizers.AdamW(0.001, learning_rate=0.01)
    new_model.compile(optimizer=optimizer)
    new_model.fit(train_ds, validation_data=val_ds, epochs=5)
    
    tf.saved_model.save(new_model, prefix + "ranking_model")
    return new_model


def preprocess_ranking_train_df(events_df, items_df, config):
    if "policy_config" in config["model_configuration"]["ranking_model"].keys():
        events_df_pivoted = (
            events_df.groupBy("session_id", "item_id")
            .pivot("event_type")
            .agg(sum("event_value"))
        )
        items_scores = events_df_pivoted.withColumn(
            "score",
            expr(config["model_configuration"]["ranking_model"]["policy_config"]),
        )
    else:
        items_scores = events_df.groupBy("session_id", "item_id").agg(
            sum("event_weight").alias("score")
        )

    ranking_train_df = items_scores.join(
        events_df.select("item_id", "session_id", "longitude", "latitude", "language", "useragent").distinct(), # TODO expensive operation
        "session_id",
        "inner",
    ).join(items_df, items_df[config["product_list"]["primary_key"]] == items_scores["item_id"], "inner")
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
    retrained_model.save(model_name)
    logging.info(f"New {model_name} model version is {retrained_model.version}")
    return retrained_model.version


def update_deployment(ms, project, new_version, model_name, deployment_name):
    deployment = ms.get_deployment(deployment_name)
    deployment.model_version = new_version
    deployment.artifact_version = "CREATE"
    # deployment.script_file = os.path.join( # why is it here?
    #     "/Projects", project.name, "Resources", f"{model_name}_predictor.py"
    # ).replace("\\", "/")

    logging.info(f"New deployment properties: {deployment.to_dict()}") # explain this line of the code?

    try:
        deployment.save(await_update=120)
    except ModelServingException as e:
        logging.info(f"deployment.save(await_update=120) failed. {e}")


def update_opensearch_index(opensearch_api, config, retrieval_model, prefix):
    index_name = opensearch_api.get_project_index(config["product_list"]["feature_view_name"])
    
    items_ds = tf.data.Dataset.from_tensor_slices({col: items_df[col] for col in items_df})
    item_embeddings = items_ds.batch(2048).map(
        lambda x: (x[config['product_list']["primary_key"]], retrieval_model._candidate_model(x))
    )

    actions = ""
    for batch in item_embeddings:
        item_id_list, embedding_list = batch
        item_id_list = item_id_list.numpy().astype(int)
        embedding_list = embedding_list.numpy()

        for item_id, embedding in zip(item_id_list, embedding_list):
            actions += json.dumps({"index": {"_index": index_name, "_id": item_id, "_source": {prefix + "vector": embedding}}}) + "\n"
            actions += json.dumps({"doc_as_upsert": True}) + "\n"

    client = OpenSearch(**opensearch_api.get_default_py_config())
    client.bulk(actions)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help="Name of DE project", default="none")
    args = parser.parse_args()

    project, fs, mr, ms, prefix, config, opensearch_api = login_to_project(args)
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
    query_model, candidate_model = train_retrieval_model(df_to_ds(items_df), train_ds, val_ds, mr, prefix)
    new_v_query_model = save_model_to_registry(query_model, mr, prefix + "query_model")
    update_deployment(
        ms,
        project,
        new_v_query_model,
        prefix + "query_model",
        (prefix + "query_model").replace("_", "").lower(),
    )
    new_v_candidate_model = save_model_to_registry(candidate_model, mr, prefix + "candidate_model")
    update_opensearch_index(opensearch_api, config, candidate_model, prefix)
            
    # Ranking model
    train_ds, val_ds = preprocess_ranking_train_df(events_df, items_df, config)
    ranking_model = train_ranking_model(train_ds, val_ds, mr, prefix)
    new_version = save_model_to_registry(ranking_model, mr, prefix + "ranking_model")
    update_deployment(
        ms,
        project,
        new_version,
        prefix + "ranking_model",
        (prefix + "ranking_deployment").replace("_", "").lower(),
    )
