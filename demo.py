 # If you want run this repository, run main.py thx
import os
import torch
from PIL import Image
from glob import glob
from torchvision import transforms

def list_models(paths):
    return glob(os.path.join(paths, '*.pth'))

def list_images(paths):
    return glob(os.path.join(paths, '*.jpg'))

def demo(device, path_models, path_images):

    device = torch.device(device)

    transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

    for model in list_models(path_models):

        model = torch.load(model, map_location=device)
        model = model['model']
        model = model.to(device)
        model.eval()

        

        for img in list_images(path_images):
            
            img = Image.open(img)
            img = transform(img)
            img = img.to(device)
            img = img[None, :, :, :]

            pred = model(img)

            print(torch.exp(pred))

demo(1, './checkpoints', './images')




            



