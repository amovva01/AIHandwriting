import cv2
import torch

def getLines(img: str):
    image = cv2.imread(img)
    