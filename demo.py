 # If you want run this repository, run main.py thx
import torch

from glob import glob

def list_models(paths):
    models = glob(paths, '*.pth')
    return models

def imprimeArqs(device):

    device = torch.device(device)

    for model in list_models('./checkpoint'):
        model = torch.load(model, map_location=device)
        print(model['model'])
