import cv2
import glob
import math
import numpy as np
import os

files = glob.glob('debug_digits/*.jpeg')
if not files: exit()

n = len(files)
cols = 10
rows = math.ceil(n / cols)

canvas = np.zeros((rows * 28, cols * 28), dtype=np.uint8)

for i, f in enumerate(files):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    r = i // cols
    c = i % cols
    canvas[r*28:(r+1)*28, c*28:(c+1)*28] = img

cv2.imwrite(r'C:\Users\ghodv\.gemini\antigravity\brain\2d4788d0-120e-4fc9-a97f-64d699a008b9\digit_crops.jpg', canvas)
print("Saved to digit_crops.jpg")
