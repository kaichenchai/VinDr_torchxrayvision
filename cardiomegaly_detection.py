import argparse
import os

import numpy as np
import skimage
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
    
    if os.path.isdir(args.path):
        files = os.listdir(args.path)
        file_paths = [os.path.join(args.path, file) for file in files if file.endswith(".png")]
        imgs = [skimage.io.imread(file_path) for file_path in file_paths]
    else:
        file_paths = [os.path.split(args.path)[1]]
        imgs = [skimage.io.imread(args.path)]
        
    print(len(imgs))

    model = xrv.baseline_models.chestx_det.PSPNet()
    targets = list(model.targets)
    heart_index = targets.index("Heart")
    l_lung_index = targets.index("Left Lung")
    r_lung_index = targets.index("Right Lung")
    diaphragm_index = targets.index("Facies Diaphragmatica")
    
    for img, filename in zip(imgs, file_paths):
        img = image_to_torch(img)
        with torch.no_grad():
            pred = model(img)
                
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        
        heart_width = 0
        
        for col in range(pred[0, heart_index].shape[1]-1, -1, -1):
            if pred[0, heart_index][col].sum() > 0:
                heart_width = col
                break

        for col in range(pred[0, heart_index].shape[1]):
            if pred[0, heart_index][col].sum() > 0:
                heart_width -= col
                break

        lungs_width = 0
        
        #r lung rightmost
        for col in range(pred[0, r_lung_index].shape[1]-1, -1, -1):
            if pred[0, r_lung_index][col].sum() > 0:
                lungs_width = col
                break

        #l lung leftmost 
        for col in range(pred[0, l_lung_index].shape[1]):
            if pred[0, r_lung_index][col].sum() > 0:
                lungs_width -= col
                break
            
        diaphragm_width = 0
        
        for col in range(pred[0, diaphragm_index].shape[1]-1, -1, -1):
            if pred[0, diaphragm_index][col].sum() > 0:
                diaphragm_width = col
                break
            
        for col in range(pred[0, diaphragm_index].shape[1]):
            if pred[0, diaphragm_index][col].sum() > 0:
                diaphragm_width -= col
                break
        
        cardiomegaly = (heart_width >= 0.5 * lungs_width)
        # cardiomegaly = (heart_width >= 0.5 * lungs_width) or (heart_width >= 0.5 * diaphragm_width)
        
        print(f"""File name: {filename}
        Lungs width: {lungs_width}
        Heart width: {heart_width}
        Diaphragm width: {diaphragm_width}
        Cardiomegaly: {cardiomegaly}""")
    
    """fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(img[0])
    axs[0,1].imshow(pred[0,heart_index])
    axs[1,0].imshow(pred[0,l_lung_index])
    axs[1,1].imshow(pred[0,r_lung_index])
    plt.show()"""