# The job retraining the ranking model. Compares the size of current training dataset and "observations" FG.
# If diff > 10%, creates new training dataset, retrains ranking model and updates deployment.

import hopsworks
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib
import json
import os
import logging
import sys

project = hopsworks.login()

fs = project.get_feature_store()
mr = project.get_model_registry()

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

td_test_size = config['model_configuration']['ranking_model']['test_size']

observations_fg = fs.get_feature_group(name=prefix + "observations", version=1)

try:
    fg_stat = observations_fg.statistics
except Exception as e:
    logging.info(f"Couldn't retrieve observations FG statistics. Quitting. Error: {e}")
    # no data available yet - no observations received
    sys.exit()

fg_size = next(column['count'] for column in fg_stat.to_dict()['featureDescriptiveStatistics'] if column['featureName'] == 'request_id') 
logging.info(f"Observations feature group size is {fg_size} rows")

training_fv = fs.get_feature_view(name=prefix + "training")
td_list = training_fv.get_training_datasets()

if not td_list:
    # todo is there any scenario when this happens?
    logging.info("No training datasets exist. Creating a new version.")
    td_version, _ = training_fv.create_train_test_split(test_size=td_test_size, description='Ranking model training dataset',
                                                        write_options={"wait_for_job": True})
else:
    td_version = max(training_fv.get_training_datasets(), key=lambda m: m.version).version
    logging.info(f"Retrieving training dataset of version {td_version}")

training_ds = training_fv._feature_view_engine._get_training_dataset_metadata(
    feature_view_obj=training_fv,
    training_dataset_version=td_version,
)

ds_size = 0  # training dataset size

train_test_stats = [m.feature_descriptive_statistics for m in training_fv.get_training_dataset_statistics(td_version).split_statistics]
for feat_list in train_test_stats:
    for feature in feat_list:
        if feature.to_dict()['featureName'] == 'request_id':
            ds_size += feature.to_dict()['count']

logging.info(f"Training dataset size is {ds_size} rows")
            
# Comparing data
size_ratio = fg_size / ds_size

if not size_ratio or size_ratio < config['model_configuration']['ranking_model']['retrain_ratio_threshold']:
    logging.info("The new data is less than ratio threshold of the current dataset size. Quitting.")
    sys.exit()

logging.info("The new data is more than ratio threshold of the current dataset size. Starting model retraining.")

# Create new training dataset version
td_version, _ = training_fv.create_train_test_split(test_size=td_test_size, description='Ranking model training dataset',
                                                    write_options= {"wait_for_job": True})
X_train, X_test, y_train, y_test = training_fv.get_train_test_split(training_dataset_version=td_version)

cat_features = list(
    X_train.select_dtypes(include=['string', 'object']).columns
)
exclude_ids = [col for col in X_train.columns if "id" in col] # todo checking on occurance of string "id" is fucked
X_train.drop(exclude_ids, axis=1, inplace=True)
X_test.drop(exclude_ids, axis=1, inplace=True)

pool_train = Pool(X_train, y_train, cat_features=cat_features)
pool_val = Pool(X_test, y_test, cat_features=cat_features)

if config['model_configuration']['ranking_model']['model'] == "CatBoostClassifier":
    model = CatBoostClassifier(
        depth=10,
        scale_pos_weight=10,
        early_stopping_rounds=5,
        use_best_model=True,
        **config['model_configuration']['ranking_model']['model_parameters']
    )
else:
    raise ValueError("Invalid ranking model specified in config")

model.fit(pool_train, eval_set=pool_val)

joblib.dump(model, 'ranking_model.pkl')

# connect to Hopsworks Model Registry
mr = project.get_model_registry()

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

input_example = X_train.sample().to_dict("records")
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

ranking_model = mr.python.create_model(
    name=prefix + "ranking_model",
    model_schema=model_schema,
    input_example=input_example,
    description="Ranking model that scores item candidates",
)
ranking_model.save("ranking_model.pkl")

logging.info(f"New model version is {ranking_model.version}")

# Updating deployment
ms = project.get_model_serving()
deployment = ms.get_deployment(prefix + "ranking_deployment")
deployment.model_version = ranking_model.version
deployment.artifact_version = "CREATE"
deployment.script_file = os.path.join("/Projects", project.name, "Resources", 'ranking_model_predictor.py').replace('\\', '/')  # hardcode filename due to the bug

logging.info(f"New deployment properties: {deployment.to_dict()}")

try:
    deployment.save(await_update=120)
except ModelServingException as e:
    logging.info(f"deployment.save(await_update=120) failed. {e}")