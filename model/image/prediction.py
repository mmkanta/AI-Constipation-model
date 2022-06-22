import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import matplotlib.cm as cm
import os
import json
from .gradcam_generator import save_gradcam, make_gradcam_heatmap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def make_prediction(result_path, image_file):
    image_size = 300

    model = load_model(os.path.join(BASE_DIR, "image", "model_image.h5"))
    img = load_img(
        os.path.join(result_path, image_file),
        color_mode='rgb',
        target_size=(image_size, image_size)
    )
    img = np.expand_dims(img_to_array(img) / 255, axis=0)
    pred_result = model.predict(img)[0][0]

    heatmap = make_gradcam_heatmap(img, model, 'top_activation')
    cam_path = save_gradcam(os.path.join(result_path, image_file),
                            heatmap,
                            image_size,
                            os.path.join(result_path, "result", "gradcam.png"))

    with open(os.path.join(result_path, "result", "prediction.txt"), 'w') as f:
        json.dump({"DD_probability": float(pred_result)}, f)
    return pred_result, cam_path