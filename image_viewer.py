import argparse
import os

import numpy as np
import skimage
import matplotlib.pyplot as plt
import torch
import torchvision
import torchxrayvision as xrv

def image_to_torch(img: np.ndarray):
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img[None, ...] # Make single color channel
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
    img = transform(img)
    img = torch.from_numpy(img)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")    
    args = parser.parse_args()
    
    file_paths = os.path.split(args.path)[1]
    img = skimage.io.imread(args.path)
        
    model = xrv.baseline_models.chestx_det.PSPNet()
    
    img = image_to_torch(img)
    with torch.no_grad():
        pred = model(img)
            
    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    
    plt.imshow(pred[0,10])
    plt.show()
    
    """plt.figure(figsize = (26,5))
    plt.subplot(1, len(model.targets) + 1, 1)
    plt.imshow(img[0], cmap='gray')
    for i in range(len(model.targets)):
        plt.subplot(1, len(model.targets) + 1, i+2)
        plt.imshow(pred[0, i])
        plt.title(model.targets[i])
        plt.axis('off')
    plt.show()"""