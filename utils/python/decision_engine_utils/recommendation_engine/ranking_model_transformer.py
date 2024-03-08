import pandas as pd
import yaml
import hopsworks
from opensearchpy import OpenSearch


class Transformer(object):
    
    def __init__(self):            
        project = hopsworks.connection().get_project()
        fs = project.get_feature_store()
        
        dataset_api = project.get_dataset_api()
        downloaded_file_path = dataset_api.download(
            f"Resources/decision-engine/h_and_m/configuration.yml" # TODO remove hardcode - how to pass args to transformer?
        )
        with open(downloaded_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        prefix = "de_" + self.config["name"] + "_"
    
        # create opensearch client
        opensearch_api = project.get_opensearch_api()
        self.os_client = OpenSearch(**opensearch_api.get_default_py_config())
        index_name = opensearch_api.get_project_index(
            self.config['product_list']["feature_view_name"]
        )
        self.item_index = opensearch_api.get_project_index(index_name)

        self.items_fv = fs.get_feature_view(
            name=prefix + self.config['product_list']["feature_view_name"], 
            version=1,
        )
        
    def preprocess(self, inputs):
        inputs = inputs["instances"]
        
        query_emb = inputs.pop(query_emb)
        session_features = inputs
        
        neighbors = self.search_candidates(query_emb, k=100)
        item_id_list = [int(el["_id"]) for el in neighbors]
        items_df = pd.DataFrame(
            self.items_fv.get_feature_vectors(entry=[{self.config["product_list"]["primary_key"]: it_id} for it_id in item_id_list]))   
            
        return { 
            "inputs" : [{"item_features": it_feat.values.tolist(), "session_features": session_features} for it_feat in items_df]
        }

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
        return self.os_client.search(body=query, index=self.item_index)["hits"]["hits"]

    def postprocess(self, outputs):
        # Extract predictions from the outputs
        preds = outputs["predictions"]
        
        # Merge prediction scores and corresponding article IDs into a list of tuples
        ranking = list(zip(preds["scores"], preds[self.config["product_list"]["primary_key"]]))
        
        # TODO add item features again (for frontend to easily show items properties)
        # TODO add decisions FG logging
        
        # Sort the ranking list by score in descending order
        ranking.sort(reverse=True)
        
        # Return the sorted ranking list
        return { 
            "ranking": ranking,
        }