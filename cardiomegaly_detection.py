import argparse
import os

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
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
    parser.add_argument("--path", type=str, required=True)    
    parser.add_argument("--output_file", type=str, required=False)
    parser.add_argument("--show_plots", type=bool, default=False, required=False)
    parser.add_argument("--verbose", type=bool, default=False, required=False)
    parser.add_argument("--threshold", type=float, default=0.5, required=False)
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        file_names = os.listdir(args.path)
        file_paths = [os.path.join(args.path, file) for file in file_names if file.endswith(".png")]
        imgs = [skimage.io.imread(file_path) for file_path in file_paths]
    else:
        file_names = [os.path.split(args.path)[1]]
        imgs = [skimage.io.imread(args.path)]
        
    print(f"Parsing files: {len(imgs)}")

    model = xrv.baseline_models.chestx_det.PSPNet()
    targets = list(model.targets)
    heart_index = targets.index("Heart")
    l_lung_index = targets.index("Left Lung")
    r_lung_index = targets.index("Right Lung")
    diaphragm_index = targets.index("Facies Diaphragmatica")
    
    output_list = []
    
    for img, filename in tqdm(zip(imgs, file_names), total=len(imgs)):
        img = image_to_torch(img)
        with torch.no_grad():
            pred = model(img)
                
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        
        heart_width = 0
        
        #heart width
        for col in range(pred[0, heart_index].shape[1]-1, -1, -1):
            if pred[0, heart_index][:,col].sum() > 0:
                heart_width = col
                break

        for col in range(pred[0, heart_index].shape[1]):
            if pred[0, heart_index][:,col].sum() > 0:
                heart_width -= col
                break
        
        if args.show_plots:
            plt.imshow(pred[0, heart_index])
            plt.show()

        lungs_width = 0
        
        #r lung rightmost (on right side of x-ray)
        for col in range(pred[0, l_lung_index].shape[1]-1, -1, -1):
            if pred[0, l_lung_index][:,col].sum() > 0:
                lungs_width = col
                break

        #r lung leftmost (on left side of x-ray)
        for col in range(pred[0, r_lung_index].shape[1]):
            if pred[0, r_lung_index][:,col].sum() > 0:
                lungs_width -= col
                break

        if args.show_plots:
            plt.imshow((pred[0, l_lung_index] + pred[0, r_lung_index]))
            plt.show()
            
        diaphragm_width = 0
        
        for col in range(pred[0, diaphragm_index].shape[1]-1, -1, -1):
            if pred[0, diaphragm_index][:,col].sum() > 0:
                diaphragm_width = col
                break
            
        for col in range(pred[0, diaphragm_index].shape[1]):
            if pred[0, diaphragm_index][:,col].sum() > 0:
                diaphragm_width = diaphragm_width - col
                break
        
        if args.show_plots:
            plt.imshow(pred[0, diaphragm_index])
            plt.show()
        
        comparison = max(lungs_width, diaphragm_width)
        cardiomegaly = (heart_width >= args.threshold * comparison)
        
        if args.verbose:
            print(f"""File name: {filename}\nLungs width: {lungs_width}\nHeart width: {heart_width}\nDiaphragm width: {diaphragm_width}\nCardiomegaly: {cardiomegaly}""")
        
        if args.output_file:
            output_list.append([filename.split(".")[0],
                                heart_width,
                                lungs_width,
                                diaphragm_width,
                                cardiomegaly])
    
    if args.output_file:
        df = pd.DataFrame(output_list, columns=["file_name", "heart_width", "lungs_width", "diaphragm_width", "cardiomegaly"])
        df.to_csv(args.output_file, index = False)