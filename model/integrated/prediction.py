import json
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
from .XAI import visualize_gradcam_shap
from .model_class import IntegratedModel

MODEL_VERSION = "2"

# https://github.com/pytorch/pytorch/issues/3678
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# integrated_model = torch.load(os.path.join(BASE_DIR, "integrated2", 'multi_modal_b3.pt'), map_location=device)

integrated_model = IntegratedModel()
integrated_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "integrated", 'model_integrated2.pth'), map_location=device))

image_size = 384  
test_transforms = transforms.Compose([transforms.Resize((image_size,image_size)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   

class IntegratedDataset(Dataset):
    def __init__(self, image_paths, tabular, transform=False):
        self.image_paths = image_paths
        self.tabular = tabular
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = Image.fromarray(image, 'RGB')
        tabular = self.tabular[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, tabular

def test_model(model, loader):
    prob = []
    model.eval()
    with torch.no_grad():
        for images, tabular in loader:
            images = images.to(device).float()
            tabular = tabular.to(device).float()
            outputs = model(images, tabular)
            prob.extend(outputs.data.cpu().numpy())
    return prob
  
def make_prediction(result_path, image_file, questionnaire: list):
    img_path = os.path.join(result_path, image_file)
    cam_path = os.path.join(result_path, "result", "gradcam.png")

    prep_questionnaire = np.array(questionnaire).reshape(1,15)

    #Dataset
    testset = IntegratedDataset(np.array([img_path]), prep_questionnaire, test_transforms)
    #DataLoader
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    prediction = test_model(integrated_model, test_loader)

    for images, tabular in test_loader:
        images = images.to(device).float()
        tabular = tabular.to(device).float()
        visualize_gradcam_shap(images, tabular, integrated_model, cam_path)
        break

    # save prediction
    with open(os.path.join(result_path, "result", "prediction.txt"), 'w') as f:
        json.dump({"DD_probability": float(prediction[0]), "version": MODEL_VERSION}, f)

    return prediction, cam_path