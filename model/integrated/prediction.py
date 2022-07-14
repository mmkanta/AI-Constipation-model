import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.cm as cm
import os
import json
from sklearn.preprocessing import StandardScaler
from ..image.prediction import make_gradcam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image_size = 300

model = load_model(os.path.join(BASE_DIR, "integrated", '2_model_integrated.h5'))
efnb1 = EfficientNetB3(weights='imagenet', include_top=False)
for layer in efnb1.layers:
  layer.trainable = False

def extract_features(x):
  features = efnb1.predict(x)
  features = Flatten()(features)

  features = Dense(1024, activation='relu')(features)
  output_features = Dense(64, activation='relu')(features)

  return output_features

def make_prediction(result_path, image_file, questionnaire):
    img_path = os.path.join(result_path, image_file)
    cam_path = os.path.join(result_path, "result", "gradcam.png")

    # load + prepare image
    img = load_img(
        img_path,
        color_mode='rgb',
        target_size=(image_size, image_size)
    )
    img = np.expand_dims(img_to_array(img), axis=0)
    img = tf.convert_to_tensor(img)

    # extract features from image
    feat = extract_features(img)
    
    # get model's prediction
    pred_result = model.predict([feat, tf.convert_to_tensor([questionnaire])])[0][0]

    # create + save gradcam
    make_gradcam(img/255, img_path, image_size, cam_path)

    # save prediction
    with open(os.path.join(result_path, "result", "prediction.txt"), 'w') as f:
        json.dump({"DD_probability": float(pred_result)}, f)

    return pred_result, cam_path