import hopsworks
import yaml

class Transformer(object):
    
    def __init__(self):            
        self.project = hopsworks.connection().get_project()
        ms = self.project.get_model_serving()
        
        dataset_api = self.project.get_dataset_api()
        downloaded_file_path = dataset_api.download(
            "Resources/decision-engine/h_and_m/configuration.yml" # TODO remove hardcode - how to pass args to transformer?
        )
        with open(downloaded_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prefix = "de_" + self.config["name"] + "_"
    
        self.ranking_server = ms.get_deployment((self.prefix + "ranking_deployment").replace("_", "").lower())
        self.inputs = None
        
    def preprocess(self, inputs): 
        self.inputs = inputs["instances"][0][0]
        
        context_item_ids = self.inputs.pop('context_item_ids')
        if not context_item_ids:
            fs = self.project.get_feature_store()
            items_fg = fs.get_feature_group(
                name=self.prefix + self.config['product_list']["feature_view_name"], 
                version=1,
            )
            pk_col = self.config['product_list']['primary_key']
            context_item_ids = items_fg.select([pk_col]).show(5)[pk_col].tolist()

        model_input = [{'context_item_ids': list(map(str, context_item_ids))}]
        print("model input: ", model_input)
        return model_input
    
    def postprocess(self, outputs):
        print("model output: ", outputs)
        # Return ordered ranking predictions 
        model_input = { "instances": [[{"query_emb": outputs[0]} | self.inputs]]} 
        print('ranking model input: ', model_input)      
        return {
            "predictions": self.ranking_server.predict(model_input),
        }