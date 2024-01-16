import os
from PIL import Image as img
import matplotlib.pyplot as plt
import numpy as np
import cv2


LIST_PATH = [
    "C:/Users/Dimit/Documents/train",
    "C:/Users/Dimit/Pictures/Soutenance",
    "C:/Users/Dimit/Pictures/DatasetRoute/DataTps",
    "C:/Users/Dimit/Pictures/Carte_noel",
    "C:/Users/Dimit/Pictures/Screenshots",
    "C:/Users/Dimit/kaggletitanic/monet_jpg",
    "C:/Users/Dimit/kaggletitanic/photo_jpg"
]
# print(LIST_PATH)

def gaussianNoise (image):
    gauss = np.random.normal(0,1,image.shape[0]*image.shape[1]*image.shape[2]*image.shape[3])
    gauss = np.reshape(gauss,image.shape).astype("uint8")
    image_noise_gauss = cv2.add(image,gauss)
    return image_noise_gauss  

def speckleNoise (image):
    gauss = np.random.normal(0,1,image.shape[0]*image.shape[1]*image.shape[2]*image.shape[3])
    gauss = np.reshape(gauss,image.shape).astype("uint8")
    image_noise_gauss = image + image*gauss
    return image_noise_gauss

def generate_random_value(batch_size):
    resulut = np.zeros((batch_size,256,256,3)).astype("uint8")
    l = ["png","jpg","jpeg"]
    for i in range (batch_size):
        random_path = np.random.randint(len(LIST_PATH))
        liste = os.listdir(LIST_PATH[random_path])
        random_value = np.random.randint(len(liste))
        while liste[random_value].split('.')[-1].lower() not in l :
            random_value = np.random.randint(len(liste))
        image = img.open(f"{LIST_PATH[random_path]}/{liste[random_value]}").convert("RGB").resize((256,256))
        image = np.array(image)
        resulut[i] = image      
    return resulut
def generate_random_noise_value(result,batch_size):
    res = result.copy()
    res[:batch_size//2] = gaussianNoise(res[:batch_size//2])
    res[batch_size//2:] = gaussianNoise(res[batch_size//2:])
    return (res.astype("float64")-127.5)/127.5
    
def resize_value (value,batch):
    # resutlt= np.zeros((batch,256,256,3))
    # for i in range (batch):
    #     resutlt[i] = cv2.resize(value[i],(256,256))
    return (value.astype("float64")-127.5)/127.5




