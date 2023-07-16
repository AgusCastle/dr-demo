import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import torch
from torchvision import transforms


def viewGradCam(model, tipo:str , img_path: str, device : int = 0):
    
    model.eval()
    trans = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                ])
    
    img = Image.open(img_path)

    img = trans(img)


    img = torch.permute(img, (1,2,0))
    
    img = np.array(img) 
    #img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    input_tensor = input_tensor.to(device)

    pred = model(input_tensor)
    target = int(torch.argmax(pred))

    targets = [ClassifierOutputTarget(target)]

    if tipo == 'convnext_0000':
        target_layers = [model.features]
    elif tipo == 'convnext_0001':
        target_layers = [model.attnblocks.cab_]
    else:
        target_layers = [model.attb.cab_]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])

    if tipo in ['convnext_0000', 'convnext_0001']:
        camd = np.uint8(255*grayscale_cams[0, :])
        camd = cv2.merge([camd, camd, camd])
        images = np.hstack((np.uint8(255*img), camd , cam_image))
        Image.fromarray(images).show()
        input()
    
    return pred, cam

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)