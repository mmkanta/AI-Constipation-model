import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import matplotlib.cm as cm
import os
import json
from .gradcam_generator import save_gradcam, make_gradcam_heatmap
from efficientnet.tfkeras import EfficientNetB3
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = load_model(os.path.join(BASE_DIR, "image", "model_image.h5"))

def make_gradcam(img, img_path, image_size, cam_path):
    heatmap = make_gradcam_heatmap(img, model, 'top_activation')

    heatmap = cv2.resize(heatmap, (300, 300), cv2.INTER_LINEAR)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / heatmap.max()
    
    _ = save_gradcam(img_path,
                    heatmap,
                    image_size,
                    cam_path)
    return
    
def make_prediction(result_path, image_file):
    image_size = 300

    img_path = os.path.join(result_path, image_file)
    cam_path = os.path.join(result_path, "result", "gradcam.png")

    # load + prepare image
    img = load_img(
        img_path,
        color_mode='rgb',
        target_size=(image_size, image_size)
    )
    img = np.expand_dims(img_to_array(img) / 255, axis=0)

    # get model's prediction
    pred_result = model.predict(img)[0][0]

    # create + save gradcam
    make_gradcam(img, img_path, image_size, cam_path)

    # save prediction
    with open(os.path.join(result_path, "result", "prediction.txt"), 'w') as f:
        json.dump({"DD_probability": float(pred_result)}, f)
    return pred_result, cam_path