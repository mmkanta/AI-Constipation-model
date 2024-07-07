import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

denormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                  transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])


def visualize_gradcam(input_tensor, model, cam_path):

    input_img_ = denormalize(input_tensor)
    input_img_show = torchvision.transforms.ToPILImage()(input_img_.squeeze(0))
    input_img_array = np.array(input_img_show)
    input_img_array = np.float32(input_img_array) / 255
    
    model.eval()
    with torch.no_grad():
        scores = model(input_tensor)
        _, predicted = torch.max(scores, 1)
    targets = [ClassifierOutputTarget(predicted)] 
    target_layers = [model.features[-1]]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
        cam_image = show_cam_on_image(input_img_array, grayscale_cams[0, :], use_rgb=True)

    Image.fromarray(cam_image).save(cam_path, "PNG")