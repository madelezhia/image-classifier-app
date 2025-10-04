"""
This module performs image classification using a pre-trained ResNet18.
"""

import torch
from torchvision import models, transforms
from PIL import Image

# Charger le modele pre-entraine
model = models.resnet18(pretrained=True)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Labels ImageNet
LABELS = models.ResNet18_Weights.DEFAULT.meta["categories"]

def predict(image_path):
    """
    Classify the image at the given path.
    Returns a dict with top 5 class probabilities.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5 = torch.topk(probs, 5)
        return {LABELS[i]: float(probs[i]) for i in top5.indices}
    except Exception:
        return None