import os
import joblib
import numpy as np
import pandas as pd

import logging

class Predict(object):
    
    def __init__(self):
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/ranking_model.pkl")

    def predict(self, inputs):
        features_df = pd.DataFrame(inputs[0])
    
        for column in features_df.columns:
            if features_df[column].dtype == 'object':
                features_df[column] = features_df[column].astype('category')
    
        logging.info("predict -> " + str(features_df))

        scores = self.model.predict(features_df).tolist()
        print("scores: ", scores)        

        return {
            "score": scores, 
        }