# The job for consuming observations from Kafka topic. Runs on schedule, inserts stream into observations FG. 
# On the first run, autodetects event schema and creates "observations" FG, "training" FV and empty training dataset.

# naming conventions here are fucked tbh
import logging
import hopsworks
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, BooleanType
from pyspark.sql.functions import from_json
from hsfs.feature import Feature

# Setting up argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name of DE project", default='none')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# Login to Hopsworks project
project = hopsworks.login()
fs = project.get_feature_store()

# todo this part should be used after backend is ready:
# decision_engine_api = project.get_decision_engine_api()
# de_project = decision_engine_api.get_decision_engine(args.name)
# prefix = de_project.prefix
# config = de_project.config

# instead of this part:
import yaml
dataset_api = project.get_dataset_api()
downloaded_file_path = dataset_api.download("Resources/test_config.yml")  # todo remove hardcode
with open(downloaded_file_path, "r") as f:
    config = yaml.safe_load(f)
prefix = 'de_' + config['name'] + '_'

labels = config['observations']['labels']
context_features = config['observations']['features']
num_items = config['observations']['num_last_visited_items']

try:
    training_fv = fs.get_feature_view(name=prefix + "training")
except Exception as e:
    logging.info("Couldn't retrieve training FV. Creating observations FG, training FV and training dataset.")
    logging.error(e)

    def setup_observations_fg(fs):
        observations_fg = fs.get_or_create_feature_group(
            name=prefix + 'observations',
            version=1,
            description="Target data: observations received from the website",
            primary_key=['request_id'],
            stream=True,
            online_enabled=True
        )

        observation_features = [
            Feature(name='request_id', type='bigint', description='Session surrogate key'),
            Feature(name='item_p_id', type='bigint', description='Predicted item'),
            Feature(name=labels[0], type='boolean', description='Scoring value, 0 or 1'),
        ]
        # Dynamically add context features
        for feat_name in context_features:
            observation_features.append(Feature(name=feat_name, type='string',
                                                description=f'App context provided in request'))
        # Dynamically add item ID features
        for i in range(num_items):
            observation_features.append(Feature(name=f'item_{i}_id', type='bigint',
                                                description=f'Top {i + 1} previously interacted item'))

        observations_fg.save(features=observation_features)

        return observations_fg

    observations_fg = setup_observations_fg(fs)

    # workaround for self-join:
    original_fg = fs.get_feature_group(prefix + 'items')
    original_df = original_fg.read()
    items_fgs = [original_fg]
    for i in range(num_items):
        copy_fg = fs.create_feature_group(prefix + f'items__copy_{i}',
                                          description='COPY Catalog for the Decision Engine project',
                                          primary_key=config['catalog']['primary_key'],
                                          online_enabled=True,
                                          version=1
                                          )
        copy_fg.insert(original_df)
        items_fgs.append(copy_fg)

    def setup_training_fv(fs, items_fgs, observations_fg):
        query = observations_fg.select_all()

        for i, items_fg in enumerate(items_fgs[:-1]):
            query = query.join(items_fg.select_all(), left_on=f"item_{i}_id", right_on="item_id", join_type="left",
                               prefix=f"it_{i}_")

        query = query.join(items_fgs[-1].select_all(), left_on="item_p_id", right_on="item_id", join_type="left",
                           prefix="it_p_")

        logging.info(query.to_string())
        training_fv = fs.get_or_create_feature_view(
            name=prefix + 'training',
            query=query,
            version=1,
            labels=labels,
        )

        return training_fv

    training_fv = setup_training_fv(fs, items_fgs, observations_fg)

    ts = training_fv.create_train_test_split(
        test_size=config['model_configuration']['ranking_model']['test_size'],
        description='Ranking model training dataset',
    )

# HACK: workaround until fix is merged
from hsfs import engine

kafka_config = engine.get_instance()._get_kafka_config(feature_store_id=fs.id)
topic_name = '_'.join([project.name, config['name'], "logObservations"])

logging.info("Creating Spark df")
df_read = spark \
    .readStream \
    .format("kafka") \
    .options(**kafka_config) \
    .option("startingOffsets", "latest") \
    .option("subscribe", topic_name) \
    .load()

# Define schema to parse by pyspark
struct_fields = [
    StructField('request_id', LongType(), True),
    StructField('item_p_id', LongType(), True),
    StructField('score', BooleanType(), True)
]
# Dynamically add context features
for feat_name in context_features:
    struct_fields.append(StructField(feat_name, StringType(), True))
# Dynamically add item ID features
for i in range(num_items):
    struct_fields.append(StructField(f'item_{i}_id', LongType(), True))

parse_schema = StructType(struct_fields)

# Deserialise data from and create streaming query
df_deser = df_read.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", parse_schema).alias("value")) \
    .select("value.request_id", "value.item_p_id", f"value.{labels[0]}",
            *[f"value.item_{i}_id" for i in range(num_items)],
            *[f"value.{feat_name}" for feat_name in context_features]) \
    .selectExpr("CAST(request_id as long)", "CAST(item_p_id as long)", f"CAST({labels[0]} as boolean)",
                *[f"CAST(item_{i}_id as long)" for i in range(num_items)],
                *[f"CAST({feat_name} as string)" for feat_name in context_features])

observations_fg = fs.get_feature_group(prefix + "observations")
fg_stream_query = observations_fg.insert_stream(df_deser)
