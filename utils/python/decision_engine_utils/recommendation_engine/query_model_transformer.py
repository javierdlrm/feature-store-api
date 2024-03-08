import hopsworks
import yaml


class Transformer(object):
    
    def __init__(self):            
        project = hopsworks.connection().get_project()
        ms = project.get_model_serving()
        
        dataset_api = project.get_dataset_api()
        downloaded_file_path = dataset_api.download(
            f"Resources/decision-engine/h_and_m/configuration.yml" # TODO remove hardcode - how to pass args to transformer?
        )
        with open(downloaded_file_path, "r") as f:
            config = yaml.safe_load(f)
        prefix = "de_" + config["name"] + "_"
    
        self.ranking_server = ms.get_deployment((prefix + "ranking_deployment").replace("_", "").lower())
        self.inputs = None
        
    def preprocess(self, inputs): # TODO cold start problem - retrieval needs to send back random items
        self.inputs = inputs["instances"] if "instances" in inputs else inputs  
        context_item_ids = self.inputs.pop("context_item_ids")
        return {
            "instances" : [context_item_ids]
        }
    
    def postprocess(self, outputs):
        # Return ordered ranking predictions        
        return {
            "predictions": self.ranking_server.predict({ "instances": {"query_emb": outputs["predictions"]} | self.inputs}),
        }