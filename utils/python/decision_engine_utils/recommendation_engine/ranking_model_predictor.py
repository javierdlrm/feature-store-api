import os
import joblib
import logging
import hopsworks


class Predict(object):

    def __init__(self):

        project = hopsworks.connection().get_project()

        mr = project.get_model_registry()
        ms = project.get_model_serving()

        # todo we should use decision_engine_api here, but it needs backend first
        import yaml
        dataset_api = project.get_dataset_api()
        downloaded_file_path = dataset_api.download("Resources/test_config.yml")
        with open(downloaded_file_path, "r") as f:
            config = yaml.safe_load(f)
        prefix = 'de_' + config['name'] + '_'

        deployment = ms.get_deployment((prefix + "ranking_deployment").replace("_", "").lower())

        try:
            model_path = os.path.join(os.environ.get("ARTIFACT_FILES_PATH", ""), "ranking_model.pkl")
            self.model = joblib.load(model_path)

            model = mr.get_model(prefix + "ranking_model", deployment.model_version)
            input_schema = model.model_schema["input_schema"]["columnar_schema"]
            self.ranking_model_feature_names = [feat["name"] for feat in input_schema]

        except Exception as e:
            logging.error(f"Ranking model file couldn't get loaded: {e}")
            self.model = None

    def predict(self, inputs):
        item_ids = inputs[0].pop("item_ids")

        if self.model is not None:
            features = inputs[0].pop("ranking_features")
            filtered_features = {key: value for key, value in features.items() if
                                 key in self.ranking_model_feature_names}
            scores = self.model.predict_proba(filtered_features)[:, 1].tolist()  # get scores of the positive class
        else:
            logging.info("Returning closest item ids with OpenSearch similarity scores.")
            scores = inputs[0].pop("os_scores")

        return {"scores": scores, "item_ids": item_ids}
