import numpy as np
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv

model = xrv.baseline_models.chestx_det.PSPNet()
img = skimage.io.imread("test_img.png")
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img[None, ...] # Make single color channel
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
img = transform(img)
img = torch.from_numpy(img)

with torch.no_grad():
    pred = model(img)


