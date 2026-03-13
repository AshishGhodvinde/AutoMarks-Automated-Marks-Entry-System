import cv2
import glob
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

for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img)
    tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = mnist_model(tensor)
        probabilities = F.softmax(outputs.data, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        
        print(f"File: {f} -> Predicted: {predicted.item()}, Conf: {max_prob.item():.4f}")
