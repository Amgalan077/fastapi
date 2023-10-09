import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import json

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_classes():
    dict_ = {0: 'buildings',
        1: 'forest',
        2: 'glacier',
        3: 'mountain',
        4: 'sea',
        5: 'street'}
    return dict_

def class_id_to_label(i):
    labels = load_classes()
    return labels[i]

def load_model():
    model = resnet50()
    model.fc = torch.nn.Linear(in_features=2048, out_features=6)
    model.load_state_dict(torch.load('utils/savemodel.pt', map_location='cuda'))
    model.eval()
    return model

def transorm_image(img):
    trans_image = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor()
        ]
    )
    return trans_image(img)
