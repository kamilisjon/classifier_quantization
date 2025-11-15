# image_utils.py
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import PIL.Image as pil_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# timm style ImageNet pre-process

def load_and_preprocess(image_path: Path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=pil_image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0).numpy()