import os
import cv2
import numpy as np
import random
import time

def addBG(imgPath, bgPath, threshold=70, targetColor=[255,255,255]):
    imgBG = cv2.imread(bgPath)
    img = cv2.imread(imgPath) 

    # Make them same size
    imgBG = cv2.resize(imgBG, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    target = np.array(targetColor)

    # Create Mask where value of pixel is true when diffrence is less than threshold
    mask = np.all(np.abs(img - target) <= threshold, axis=2)
    # set all pixels where mask value is true to the pixels of bg image where mask value is true
    img[mask] = imgBG[mask]
    
    cv2.imwrite(imgPath, img)




imageFolder = input('-> Image Folder: ')
bgFolder = input('-> Background Image Folder: ')


print(f'-> Adding Background... ({len(os.listdir(imageFolder))*0.01}s)')


st = time.time()

for img in os.listdir(imageFolder):
    img = f'{imageFolder}\\{img}'
    bg = f'{bgFolder}\\{random.choice(os.listdir(bgFolder))}'
    addBG(imgPath=img, bgPath=bg)

et = time.time()
print(f'-> Adding Background took {round(et-st, 2)}s')