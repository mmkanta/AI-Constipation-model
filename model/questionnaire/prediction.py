import numpy as np
import os
from catboost import CatBoostClassifier

MODEL_VERSION = "2"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "questionnaire", "model.cbm")

model = CatBoostClassifier()
model.load_model(model_path)

def make_prediction(questionnaire):
    result = model.predict(np.array(questionnaire).reshape(1, 66))[0]
    return result, MODEL_VERSION