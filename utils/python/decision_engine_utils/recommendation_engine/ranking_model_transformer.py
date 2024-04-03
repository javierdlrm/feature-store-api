import pandas as pd
import numpy as np
import yaml
import hopsworks
from opensearchpy import OpenSearch


class Transformer(object):
    
    def __init__(self):            
        project = hopsworks.connection().get_project()
        fs = project.get_feature_store()
        
        dataset_api = project.get_dataset_api()
        downloaded_file_path = dataset_api.download(
            f"Resources/decision-engine/h_and_m/configuration.yml", overwrite=True # TODO remove hardcode - how to pass args to transformer?
        )
        with open(downloaded_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prefix = "de_" + self.config["name"] + "_"
    
        # create opensearch client
        opensearch_api = project.get_opensearch_api()
        self.os_client = OpenSearch(**opensearch_api.get_default_py_config())
        self.item_index = opensearch_api.get_project_index(
            self.config['product_list']["feature_view_name"]
        )
        self.items_fv = fs.get_feature_view(
            name=self.prefix + self.config['product_list']["feature_view_name"], 
            version=1,
        )
        self.items_data = None
        
    def preprocess(self, inputs):
        inputs = inputs["instances"][0][0]
        
        query_emb = inputs.pop('query_emb')
        session_features = inputs
        
        neighbors = self.search_candidates(query_emb, k=100)
        item_id_list = [int(el["_id"]) for el in neighbors]
        items_df = self.items_fv.get_feature_vectors(entry=[{self.config["product_list"]["primary_key"]: it_id} for it_id in item_id_list], return_type="pandas")
        
        for feat, val in self.config['product_list']["schema"].items():
            if val["type"] == "float":
                items_df[feat] = items_df[feat].astype("float32")
            if "transformation" in val.keys() and val["transformation"] == "timestamp":
                items_df[feat] = items_df[feat].astype(np.int64) // 10**9
        items_df[self.config['product_list']["primary_key"]] = items_df[self.config['product_list']["primary_key"]].astype(str)
                
        self.items_data = items_df.to_dict(orient='list')
        model_input = self.items_data | {key: [value] * items_df.shape[0] for key, value in session_features.items()}
        print("model input: ", model_input)
        return [model_input]

    def search_candidates(self, query_emb, k=100):
        query = {
            "size": k,
            "query": {
                "knn": {
                    self.prefix + "vector": {
                        "vector": query_emb,
                        "k": k
                    }
                }
            }
        }
        print(self.item_index)
        return self.os_client.search(body=query, index=self.item_index)["hits"]["hits"]

    def postprocess(self, outputs):
        print("ranking model outputs: ", outputs)
        
        flattened_outputs = [score[0] for score in outputs]
        self.items_data['score'] = flattened_outputs
        
        # TODO add item features again (for frontend to easily show items properties)
        # TODO add decisions FG logging
        
        # Return the sorted ranking list
        return self.items_data