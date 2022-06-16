import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict(questionnaire):
    model_path = os.path.join(BASE_DIR, "questionnaire", "model_questionnaire.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    result = model.predict_proba(np.array([questionnaire]))[0][1]
    return result