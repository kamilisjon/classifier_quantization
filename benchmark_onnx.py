# evaluate.py
import argparse
import os
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime

from pre_process import load_and_preprocess

IMAGENET_LABELS_FILEPATH = "imagenet_class_index.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    return parser.parse_args()

def main():
    args = parse_args()

    val_dir = args.data_dir / 'ILSVRC' / 'Data' / 'CLS-LOC' / "val"
    images = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == '.JPEG'])

    # Parse labels
    labels_csv = args.data_dir / "LOC_val_solution.csv"
    imageid_to_synsets = {}
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"]
            pred_str = row["PredictionString"].strip()
            if not pred_str:
                imageid_to_synsets[image_id] = []
                continue
            tokens = pred_str.split()
            synsets = [tokens[i] for i in range(0, len(tokens), 5)]
            imageid_to_synsets[image_id] = sorted(set(synsets))
    print(f"Parsed labels for {len(imageid_to_synsets)} images")

    # Map synsets
    with open(IMAGENET_LABELS_FILEPATH, "r") as f:
        class_idx = json.load(f)
    idx_to_label = {int(k): v[1] for k, v in class_idx.items()}
    synset_to_idx = {v[0]: int(k) for k, v in class_idx.items()}

    # Build ground_truth
    ground_truth = []
    for img_path in images:
        img_id = img_path.stem
        if img_id not in imageid_to_synsets:
            raise ValueError(f"No labels for {img_id}")
        synsets = imageid_to_synsets[img_id]
        gt_indices = sorted({synset_to_idx[s] for s in synsets if s in synset_to_idx})
        if not gt_indices:
            raise ValueError(f"No mapping for {img_id}")
        ground_truth.append(gt_indices)
    print(f"Built ground_truth for {len(ground_truth)} images")

    # ONNX session
    session_options = onnxruntime.SessionOptions()
    # 0 = VERBOSE, 1 = INFO, 2 = WARNING, 3 = ERROR, 4 = FATAL
    session_options.log_severity_level = 0  # INFO
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = str(args.model_path.parent / "calibration.flatbuffers")  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
    provider = ['TensorrtExecutionProvider'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(str(args.model_path), sess_options=session_options, providers=provider)

    # Benchmark
    correct_top1 = correct_top5 = total = 0
    for i, img_path in enumerate(images):
        input_tensor = load_and_preprocess(img_path)
        outputs = ort_session.run(None, {"x": input_tensor})[0]
        pred = np.argsort(outputs[0])[-5:][::-1]

        gt_indices = set(ground_truth[i])
        total += 1
        if pred[0] in gt_indices:
            correct_top1 += 1
        if any(p in gt_indices for p in pred):
            correct_top5 += 1

        if i < 5:
            print(f"\nImage: {img_path.name}")
            print("  GT classes:")
            for g in sorted(gt_indices):
                print(f"    idx {g:4d} | {idx_to_label.get(g, 'N/A')}")
            print("  Top-5:")
            for rank, p in enumerate(pred, 1):
                print(f"    {rank}: {p:4d} | {idx_to_label.get(p, 'N/A')}")

        if i % 1000 == 0:
            print(f"Processed {i}/{len(images)}")

    top1_acc = correct_top1 / total * 100
    top5_acc = correct_top5 / total * 100
    print(f"\nTop-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")

if __name__ == "__main__":
    main()