 # If you want run this repository, run main.py thx
import os
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
from pathlib import Path
from utils.gradcam import viewGradCam, show_cam_on_image
from models.sparse import SparseFusion
import numpy as np
import cv2
import pdb

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

    dict_gradcams = {}
    for model in list_models(path_models):
        i = model
        model = torch.load(model, map_location=device)
        model = model['model']
        model = changeLogSoftmaxbySoftmax(model, str(Path(i).stem))


        model = model.to(device)
        model.eval()

        for img in list_images(path_images):
            key = Path(img).stem
            
            gradcam = viewGradCam(model, str(Path(i).stem), 1, img, 1)
            path = img
            img = Image.open(img)
        
            img = transform(img)
            img = img.to(device)
            img = img[None, :, :, :]

            pred = model(img)

            ss = pred.tolist()[0]
            arg = int(torch.argmax(pred))
            

            print('Model {}, Prediction : {}'.format(str(Path(i).stem), arg))

            if dict_gradcams.get(key) is None:
                dict_gradcams[key] = { 'softscore': [ss], 'pred': [arg], 'cam' :[gradcam], 'path' : path }
            else:
                dict_gradcams.get(key)['softscore'].append(ss)
                dict_gradcams.get(key)['pred'].append(arg)
                dict_gradcams.get(key)['cam'].append(gradcam)

    model = torch.load('checkpoints/snfw2.pt', map_location=device)
    model = model['model']
    
    model = model.to(device)
    model.eval()

    for element in dict_gradcams.values():

        s = element['softscore']
        s = torch.permute(torch.FloatTensor(s), (1, 0)).to(device)
        s = s[None, :, :]
        pred = model(s)
        pred = int(torch.argmax(pred))
        print('SNF Prediction: {}'.format(pred))
        
        final = []
        for gradcam, pred_m in zip(element['cam'], element['pred']):
            if pred_m == pred:
                final.append(gradcam)

        dst = argmaxPlot(final)

        # dst = None
        # for i, gray in enumerate(final):    
        #     if i == 0:
        #         continue
        #     if i == 1:
        #         dst = cv2.addWeighted(final[0], 1, gray, 1, 0.0)
        #         continue
        #     dst += cv2.addWeighted(dst, 1, gray, 1, 0.0)

        img = np.array(Image.open(element['path']))
        img = cv2.resize(img, (512, 512))
        img = np.float32(img) / 255
    
        Image.fromarray(show_cam_on_image(img, cv2.normalize(dst,None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F), use_rgb=True)).show()
        
def argmaxPlot(list_grayscales):

    final_image = list_grayscales[0]

    for i in range(1, len(list_grayscales)):
        final_image = sumArgMax(final_image, list_grayscales[i])

    return final_image

def sumArgMax(ndarray1, ndarray2):
    tmp = np.zeros((512, 512))
    for i in range(512):
        for j in range(512):
            if ndarray1[i, j] > ndarray2[i, j]:
                tmp[i, j] = ndarray1[i, j]
            else:
                tmp[i, j] = ndarray2[i, j]
    return tmp
            
def changeLogSoftmaxbySoftmax(model, flag):
    if flag == 'convnext_0001':
        model.attnblocks.fc_[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    elif flag == 'convnext_0000':
        model.classifier[10] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    else:
        model.attb.fc_[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    
    return model
demo(1, './checkpoints', './images')




            



