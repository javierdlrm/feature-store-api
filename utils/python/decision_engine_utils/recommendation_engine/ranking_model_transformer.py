import pandas as pd

import hopsworks
import logging
from opensearchpy import OpenSearch
import yaml


class Transformer(object):

    def __init__(self):

        project = hopsworks.connection().get_project()

        # todo we should use decision_engine_api here, but it needs backend first
        dataset_api = project.get_dataset_api()
        downloaded_file_path = dataset_api.download("Resources/test_config.yml")
        with open(downloaded_file_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.fs = project.get_feature_store()
        self.catalog_fv = self.fs.get_feature_view(self.config['catalog']['feature_group_name'], 1)
        self.catalog_fv.init_serving()

        # create opensearch client
        opensearch_api = project.get_opensearch_api()
        self.os_client = OpenSearch(**opensearch_api.get_default_py_config())
        self.item_index = opensearch_api.get_project_index(self.config['catalog']['feature_group_name'])

    def preprocess(self, inputs):
        inputs = inputs["instances"] if "instances" in inputs else inputs
        request_id = inputs["request_id"]
        context_features = {feat: inputs[feat] for feat in self.config['observations']['features']}
        last_items_list = [inputs[f"item_{i}_id"] for i in range(self.config['observations']['num_last_visited_items'])]

        # search for candidates, the closest item for each of last visited items
        last_items_feat_vectors = \
        self.os_client.mget(index=self.item_index, body={'docs': [{'_id': it_id} for it_id in last_items_list]})['docs']
        hits = []
        for val in last_items_feat_vectors:
            try:
                embed = val['_source']['my_vector1']  # derive items embeddings from the index
            except Exception as e:
                print(f"Error while retrieving vector {val['_id']}: {e}")
                continue

            hits += self.search_candidates(embed, k=1)  # derive only top 1 closest vector

        logging.info(f"Opensearch found candidates: {hits}")

        # get features of candidate items
        pred_item_id_list = [int(el["_id"]) for el in hits]
        pred_item_os_scores_list = [el["_score"] for el in hits]
        pred_items_df = pd.DataFrame(
            self.catalog_fv.get_feature_vectors(entry=[{'item_id': it_id} for it_id in pred_item_id_list]))
        pred_items_df.rename(columns=lambda col: f'it_p_{col}', inplace=True)

        # get features of last visited items
        last_items_df = pd.DataFrame(
            self.catalog_fv.get_feature_vectors(entry=[{'item_id': it_id} for it_id in last_items_list]))

        # create input dataframe for ranking_model
        new_columns = []
        for i in range(len(last_items_df)):
            new_columns.extend([f'it_{i}_{col}' for col in last_items_df.columns])
        new_data = [item for row in last_items_df.itertuples(index=False) for item in row]
        ranking_model_input_df = pd.DataFrame([new_data], columns=new_columns).assign(**context_features)

        # merge last visited items features and candidate features for the model input
        ranking_model_input_df = ranking_model_input_df.merge(pred_items_df, how='cross')
        inputs = {"inputs": [{"item_ids": pred_item_id_list, "os_scores": pred_item_os_scores_list,
                              "ranking_features": ranking_model_input_df.to_json()}]}
        logging.info(f"Model inputs: {inputs}")
        return inputs

    def postprocess(self, outputs):
        preds = outputs["predictions"]
        ranking = list(zip(preds["scores"], preds["item_ids"]))  # merge lists
        ranking.sort(reverse=True)  # sort by score (descending)
        return {"ranking": ranking}

    def search_candidates(self, query_emb, k=100):
        query = {
            "size": k,
            "query": {
                "knn": {
                    "my_vector1": {
                        "vector": query_emb,
                        "k": k
                    }
                }
            }
        }
        return self.os_client.search(body=query, index=self.item_index)["hits"]["hits"]