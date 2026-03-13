import cv2
import glob
import os
from processor import process_image

os.makedirs('debug_digits', exist_ok=True)
images = glob.glob('images/*.jpeg')
if not images: exit()

for img_path in images:
    if 'debug_' in img_path: continue
    print(f"Processing {img_path}")
    try:
        process_image(img_path)
    except Exception as e:
        print(e)
