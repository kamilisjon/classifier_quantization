import argparse
from pathlib import Path
import shutil
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample_count", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in args.input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in {'.jpeg', '.jpg'}])
        num_to_sample = min(len(images), args.sample_count)
        sampled = random.sample(images, num_to_sample)

        for img in sampled:
            shutil.copy2(img, args.output_dir / img.name)
        print(f"Class {class_dir.name}: copied {num_to_sample}/{len(images)}")

if __name__ == "__main__":
    main()