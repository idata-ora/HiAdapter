import os
import torch
import cv2
import numpy as np
import math
from PIL import Image
import torch.fft
import pywt
from tqdm import tqdm
from skimage import data,filters,segmentation,measure,morphology,color

from histocartography.preprocessing import NucleiExtractor


from pdb import set_trace as st

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # OSError: broken data stream when reading image file (PIL bug)



# 定义 HED 矩阵
HED_MATRIX = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin 
    [0.072, 0.990, 0.105],  # Eosin 
    [0.268, 0.570, 0.776],  # DAB 
], dtype=np.float32)

def rgb_to_hed(img_rgb):
    """将BGR图像转换为HED空间"""
    img_hed = np.dot(img_rgb, HED_MATRIX.T)
    return img_hed

def OTSU(img_gray):
    assert img_gray.ndim == 2, "输入图像必须是灰度图像"
    PixSum = img_gray.size
    PixCount = np.zeros(256)
    PixRate = np.zeros(256)

    # 计算每个灰度值的像素数量
    for i in img_gray.ravel():
        PixCount[i] += 1

    # 计算每个灰度值的概率
    for j in range(256):
        PixRate[j] = PixCount[j] / PixSum

    Max_var = 0
    th = 0

    for i in range(1, 256):
        w1 = np.sum(PixRate[:i])
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            continue

        u1_tem = np.sum(np.arange(i) * PixRate[:i]) / w1
        u2_tem = np.sum(np.arange(i, 256) * PixRate[i:]) / w2
        tem_var = w1 * w2 * (u1_tem - u2_tem) ** 2

        if tem_var > Max_var:
            Max_var = tem_var
            th = i

    return th



# non SEGMENTATION
def optimized_os_extractor(img, d_kernel=4, area_thd=0):
    
    # Step0
    lower=np.array([0,0,200],dtype='uint8')
    upper=np.array([180,30,255],dtype='uint8')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # attention: bgr order
    mask=cv2.inRange(hsv,lower,upper)

    # Step1
    hed = rgb_to_hed(img) # 256,256,3
    hematoxylin = hed[:,:,0].astype(np.uint8)
    eosin = hed[:,:,1].astype(np.uint8) # 256,256
    DAB = hed[:,:,2].astype(np.uint8)


    ## HE: eosin  IHC：DAB
    thre1 = OTSU(hematoxylin)
    _, osteoid1 = cv2.threshold(hematoxylin, thre1*1.2, 255, cv2.THRESH_BINARY_INV) #0.6    0.8
    thre2 = OTSU(eosin)
    _, osteoid2 = cv2.threshold(eosin, thre2*0.8, 255, cv2.THRESH_BINARY) #1.1  1.0
    osteoid = (osteoid1.astype(bool) & osteoid2.astype(bool) & (1-mask.astype(bool))).astype(np.uint8) * 255

    
    # Step3
    img_inv = cv2.bitwise_not(osteoid)
    kernel = np.ones((d_kernel,d_kernel), np.uint8)  
    closed = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)  
    closed = cv2.bitwise_not(closed)

    h, w = img.shape[:2]
    area_thd = int(math.sqrt(w * h))
    if area_thd>0: 
        binary_mask = (closed > 0).astype(bool)
        filtered = morphology.remove_small_objects(binary_mask>0, min_size=area_thd, connectivity=2).astype(np.uint8) * 255

    final = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    return final 


def process_images(input_dir, output_dir, task):

    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
        if filename.endswith(('.png','.jpg',"jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            # print(f"Processed: {output_path}")

            img = Image.open(input_path).convert('RGB')
            img = np.array(img)  

            if task == "nuclei":
                processed_img, _ = nuclei_detector.process(img) # numpy
                processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)  
                processed_img = Image.fromarray(processed_img)  
                processed_img.save(output_path)  

            # elif task == "wave":
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            #     cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
            #     high_freq_sum = cH + cV + cD
            #     high_freq_sum = cv2.resize(high_freq_sum, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            #     processed_img = np.clip(high_freq_sum, 0, 255).astype(np.uint8)  
            #     processed_img = Image.fromarray(processed_img)  
            #     processed_img.save(output_path)  

            elif task == "non":

                processed_img = optimized_os_extractor(img)
                processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)  
                processed_img = Image.fromarray(processed_img) 
                processed_img.save(output_path)  


task = "nuclei" #nuclei non wave

if task == "nuclei":
    nuclei_detector = NucleiExtractor(pretrained_data='pannuke', batch_size=64)


input_dir = ''
output_dir = '' + task
process_images(input_dir, output_dir, task)
