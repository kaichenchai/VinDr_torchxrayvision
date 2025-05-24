import numpy as np
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv

model = xrv.baseline_models.chestx_det.PSPNet()
print(f"Model: {model} loaded")

img = skimage.io.imread("test_img.png")
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
print(img.shape)
#img = img.astype(np.uint8)
#skimage.io.imsave(fname="check_1.png", arr=img)
img = img[None, ...] # Make single color channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
#transform = torchvision.transforms.Compose([xrv.datasets.XRayResizer(512)])
img = transform(img)
#skimage.io.imsave(fname="check_2.png", arr=img)

img = torch.from_numpy(img)

with torch.no_grad():
    pred = model(img)

plt.figure(figsize = (26,5))
plt.subplot(1, len(model.targets) + 1, 1)
plt.imshow(img[0], cmap='gray')
for i in range(len(model.targets)):
    plt.subplot(1, len(model.targets) + 1, i+2)
    plt.imshow(pred[0, i])
    plt.title(model.targets[i])
    plt.axis('off')
plt.savefig("test_output.png")

pred = 1 / (1 + np.exp(-pred))  # sigmoid
pred[pred < 0.5] = 0
pred[pred > 0.5] = 1

plt.figure(figsize = (26,5))
plt.subplot(1, len(model.targets) + 1, 1)
plt.imshow(img[0], cmap='gray')
for i in range(len(model.targets)):
    plt.subplot(1, len(model.targets) + 1, i+2)
    plt.imshow(pred[0, i])
    plt.title(model.targets[i])
    plt.axis('off')
plt.savefig("test_output1.png")
