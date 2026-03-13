import cv2
import glob
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from processor import preprocess_for_mnist, mnist_model, device

files = glob.glob('debug_digits/*.jpeg')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

n = len(files)
cols = 10
rows = math.ceil(n / cols)

# 28x28 digit + text space below
cell_w = 40
cell_h = 50

canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

for i, f in enumerate(files):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img)
    tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = mnist_model(tensor)
        probabilities = F.softmax(outputs.data, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        pred_val = predicted.item()
        conf = max_prob.item()

    r = i // cols
    c = i % cols
    
    y_off = r * cell_h
    x_off = c * cell_w
    
    # Place original digit (convert to BGR for color text overlay)
    digit_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas[y_off:y_off+28, x_off:x_off+28] = digit_bgr
    
    # Overlay text
    color = (0, 255, 0) if conf >= 0.3 else (0, 0, 255)
    cv2.putText(canvas, f"{pred_val}", (x_off + 2, y_off + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite(r'C:\Users\ghodv\.gemini\antigravity\brain\2d4788d0-120e-4fc9-a97f-64d699a008b9\digit_preds.jpg', canvas)
print("Saved to digit_preds.jpg")
