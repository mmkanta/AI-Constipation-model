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
from ..image.gradcam_generator import make_gradcam_heatmap, save_gradcam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image_size = 300

def extract_features(x):
  efnb1 = EfficientNetB3(weights='imagenet', include_top=False)

  for layer in efnb1.layers:
    layer.trainable = True

  features = efnb1.predict(x)
  print(features.shape)
  features = Flatten()(features)
  #features = np.reshape(features, (len(generator), -1))

  features = Dense(1024, activation='relu')(features)
  #features = Dropout(0.2)(features)
  output_features = Dense(64, activation='relu')(features)

  return output_features

def make_prediction(result_path, image_file, questionnaire):
    model = load_model(os.path.join(BASE_DIR, "integrated", '2_model_integrated.h5'))
    scaler = StandardScaler()

    img = load_img(
        os.path.join(result_path, image_file),
        color_mode='rgb',
        target_size=(image_size, image_size)
    )
    img = np.expand_dims(img_to_array(img), axis=0)

    feat = extract_features(img)
    feat = scaler.fit_transform(feat)

    pred_result = model.predict([feat, np.array([questionnaire])])[0][0]

    image_model = load_model(os.path.join(BASE_DIR, "image", "model_image.h5"))

    heatmap = make_gradcam_heatmap(img/255, image_model, 'top_activation')
    cam_path = save_gradcam(os.path.join(result_path, image_file),
                            heatmap,
                            image_size,
                            os.path.join(result_path, "result", "gradcam.png"))

    with open(os.path.join(result_path, "result", "prediction.txt"), 'w') as f:
        json.dump({"DD_probability": float(pred_result)}, f)

    return pred_result, cam_path