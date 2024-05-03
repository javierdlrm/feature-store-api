import pandas as pd
import yaml
import hopsworks
import pandas as pd
import os


class Transformer(object):
    def __init__(self):
        self.project = hopsworks.connection().get_project()
        dataset_api = self.project.get_dataset_api()
        de_api = self.project.get_decision_engine_api()
        decision_engine = de_api.get_by_name(os.getenv('MODEL_NAME').removeprefix('de_').removesuffix('_ranking_model'))
        downloaded_file_path = dataset_api.download(decision_engine._config_file_path, overwrite=True)
        
        with open(downloaded_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prefix = "de_" + self.config["name"] + "_"
        fs = self.project.get_feature_store()
        self.decisions_fg = fs.get_feature_group(
            name=self.prefix + "decisions",
            version=1,
        )

    def preprocess(self, inputs):
        self.inputs = inputs["instances"][0][0]
        print("preproc input: ", inputs)
        
        self.items_data = self.inputs.pop('items')
        self.items_data.pop('score')
        self.de_name = self.inputs.pop('de_name')
        num_items = len(list(self.items_data.values())[0])
        model_input = {
            key: [value] * num_items for key, value in self.inputs.items()
        } | self.items_data
        print("model input: ", model_input) 
        return [model_input]

    def postprocess(self, outputs):
        print("outputs: ", outputs)
        flattened_outputs = outputs['score']
        self.items_data["score"] = flattened_outputs

        #TODO error due to restricted scope in insert
        # pk_type = self.config["product_list"]['schema'][self.config["product_list"]['primary_key']]['type']  # TODO in session_activity and predicted_items replace int to pk_type
        # decisions = {
        #     "decision_id": [self.decision_id],
        #     "session_id": [self.session_id],
        #     "session_activity": [list(map(int, self.context_item_ids)) or [0]],
        #     "predicted_items": [list(map(int, self.items_data[self.config["product_list"]["primary_key"]]))],
        # }
        # print("decisions: ", decisions)
        # df = pd.DataFrame.from_dict(decisions)
        # self.decisions_fg.insert(df) 
        return self.items_data