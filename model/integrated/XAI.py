import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from .gradcam import GradCAM, BinaryClassifierOutputTarget, show_cam_on_image
from captum.attr import DeepLiftShap
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

denormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                  transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

def calculate_shap(integrated_model, input_img, input_tab):
    img_base = torch.zeros(2, 3, 384,384, requires_grad=False).to(device).float()
    tabular_base_0 = torch.zeros(1, 15, requires_grad=False).to(device).float()
    tabular_base_1 = torch.ones(1, 15, requires_grad=False).to(device).float()
    tabular_base = torch.cat((tabular_base_0, tabular_base_1), 0)
    dl = DeepLiftShap(integrated_model)
    shap = []
    features_tab = []
    integrated_model.eval()
    with torch.no_grad():
        attr = dl.attribute((input_img, input_tab), baselines=(img_base,tabular_base), target=0)
        shap.append(attr)
        features_tab.append([input_tab.data.cpu().numpy()])
    return  shap, features_tab
        
def plot_shap(shap, features_tabular):
    
    feature_name = ['DistFreq', 'DistSev', 'DistSevFreq', 'DistDur','FreqStool', 'Incomplete', 'Strain', 'Hard', 'Block', 'Digit',
                             'BloatFreq','BloatSev','BloatSevFreq','BloatDur','SevScale']
    
    for i in range(len(feature_name)):
        feature_name[i] = feature_name[i] + ' =' + ' ' + str(int(features_tabular[i]))
    feature_importances = shap[0][1][0].data.cpu().numpy()
    deepshap = {"Features": feature_name, "SHAP":feature_importances}
    data_frame  = pd.DataFrame(data = deepshap)
    data_frame = data_frame.sort_values(by=['SHAP'], ascending=False)
    feature_importances = data_frame['SHAP'].tolist()
    feature_name = data_frame['Features'].tolist()
    
    deepshap_positive, deepshap_negative = [], []
    for i in range(len(feature_importances)):
        if feature_importances[i] >= 0 :
            deepshap_positive.append(feature_importances[i])
            deepshap_negative.append(0)
        else :
            deepshap_negative.append(feature_importances[i])
            deepshap_positive.append(0)
    return  deepshap_negative, deepshap_positive, feature_name

def visualize_gradcam_shap(input_img, input_tab, integrated_model, cam_path):

    integrated_model.eval()
    with torch.no_grad():
        scores = integrated_model(input_img, input_tab)
        predicted = torch.round(scores)
    input_img_ = denormalize(input_img)
    input_img_show = torchvision.transforms.ToPILImage()(input_img_.squeeze(0))
    input_img_array = np.array(input_img_show)
    input_img_array = np.float32(input_img_array) / 255
    targets = [BinaryClassifierOutputTarget(predicted)] # Binary
    target_layers = [integrated_model.image_model.features[-1]]
    
    with GradCAM(model=integrated_model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_img, input_tensor_tab=input_tab, targets=targets, eigen_smooth=True)
        cam_image = show_cam_on_image(input_img_array, grayscale_cams[0, :], use_rgb=True)
        
    Image.fromarray(cam_image).save(cam_path, "PNG")
    