from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
import PIL.Image as pil_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# timm style ImageNet pre-process

def load_and_preprocess(image_paths: list[Path], padd_to_batch_size: int | None = None):
    images = [Image.open(p).convert('RGB') for p in image_paths]
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=pil_image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    processed = [transform(img) for img in images]
   
    # Handle batch padding logic
    if padd_to_batch_size is not None:
        n = len(processed)
        if n > padd_to_batch_size:
            raise ValueError( f"Received {n} images, but batch size is {padd_to_batch_size}. Cannot exceed batch size.")

        if n < padd_to_batch_size:
            c, h, w = processed[0].shape
            num_pad = padd_to_batch_size - n
            pad_images = [torch.zeros((c, h, w)) for _ in range(num_pad)]
            processed.extend(pad_images)

    return torch.stack(processed).numpy()