import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import warnings
import os
import json
from .model_class_image import ImageModel
from .gradcam_image import visualize_gradcam

MODEL_VERSION = "2"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_model = ImageModel().to(device)
img_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "image", 'model_image2.pth')))

image_size = 384                                                              
test_transforms = transforms.Compose([transforms.Resize((image_size,image_size)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

class DDDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = Image.fromarray(image, 'RGB')
        
        if self.transform is not None:
            image=self.transform(image)
        
        return image
    
def test_model(model, loader):
    prediction_list = []
    model.eval()
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            prediction_list.extend(outputs.data.cpu().numpy())
    return prediction_list

def make_prediction(result_path, image_file):
    img_path = os.path.join(result_path, image_file)
    cam_path = os.path.join(result_path, "result", "gradcam.png")

    #Dataset
    testset = DDDataset(np.array([img_path]), test_transforms)
    #DataLoader
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    prediction = test_model(img_model, test_loader)

    for images in test_loader:
        images = images.to(device).float()
        visualize_gradcam(images, img_model, cam_path)
        break

    # save prediction
    with open(os.path.join(result_path, "result", "prediction.txt"), 'w') as f:
        json.dump({"DD_probability": float(prediction[0][1]), "version": MODEL_VERSION}, f)

    return prediction, cam_path
