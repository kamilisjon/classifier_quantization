import argparse
import time
from pathlib import Path
from enum import Enum
import csv
import json

import onnxruntime
import numpy as np

from pre_process import load_and_preprocess

WARMUP_RUNS_COUNT = 25
BENCHMARK_RUNS_COUNT = 100
IMAGENET_LABELS_FILEPATH = "imagenet_class_index.json"

class ExecProvider(Enum):
    CPU = 0
    CUDA = 1
    TRT_FP32 = 2
    TRT_FP16 = 3
    TRT_INT8 = 4

    def __repr__(self):
        return self.name

BENCHMARK_EXEC_PROVIDERS = [ExecProvider.CUDA, ExecProvider.TRT_FP32, ExecProvider.TRT_FP16, ExecProvider.TRT_INT8]

def setup_session(model_path: Path, exec_provider: ExecProvider) -> onnxruntime.InferenceSession:
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 0
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    match exec_provider:
        case ExecProvider.CPU: provider = ['CPUExecutionProvider']
        case ExecProvider.CUDA: provider = ['CUDAExecutionProvider']
        case ExecProvider.TRT_FP32: provider = ['TensorrtExecutionProvider']
        case ExecProvider.TRT_FP16: provider = [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
        case ExecProvider.TRT_INT8: provider = [('TensorrtExecutionProvider', 
                                                 {'trt_fp16_enable': True, 'trt_int8_enable': True,
                                                  'trt_int8_calibration_table_name': str(model_path.parent / "calibration.flatbuffers")})]
    return onnxruntime.InferenceSession(str(model_path), sess_options=session_options, providers=provider)


def benchmark_accuracy(session: onnxruntime.InferenceSession, imagenet_data_path: Path, batch_size: int):
    val_dir = imagenet_data_path / 'ILSVRC' / 'Data' / 'CLS-LOC' / "val"
    images = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == '.JPEG'])

    # Parse labels
    labels_csv = imagenet_data_path / "LOC_val_solution.csv"
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

    # Map synsets
    with open(IMAGENET_LABELS_FILEPATH, "r") as f:
        class_idx = json.load(f)
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

    # Benchmark
    correct_top1 = correct_top5 = total = 0
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch_paths = images[start:end]
        input_tensor = load_and_preprocess(batch_paths)
        outputs = session.run(None, {"x": input_tensor})[0]
        for j, out in enumerate(outputs):
            i = start + j
            pred = np.argsort(out)[-5:][::-1]
            gt_indices = set(ground_truth[i])

            total += 1
            if pred[0] in gt_indices:
                correct_top1 += 1
            if any(p in gt_indices for p in pred):
                correct_top5 += 1

            if i % 1000 == 0:
                print(f"Processed {i + 1}/{len(images)}")

    return correct_top1 / total * 100, correct_top5 / total * 100

def benchmark_speed(session: onnxruntime.InferenceSession, batch_size: int):
    input_data = np.zeros((batch_size, 3, 224, 224), np.float32)
    input_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_data, 'cuda', 0)

    iobinding = session.io_binding()
    iobinding.bind_ortvalue_input(session.get_inputs()[0].name, input_ortvalue)
    for output in session.get_outputs():
        iobinding.bind_output(output.name, 'cuda')

    # Warm up
    for _ in range(WARMUP_RUNS_COUNT):
        session.run_with_iobinding(iobinding)

    total_duration = 0.0
    for _ in range(BENCHMARK_RUNS_COUNT):
        start = time.perf_counter()
        session.run_with_iobinding(iobinding)
        batch_duration = (time.perf_counter() - start) * 1000
        total_duration += batch_duration
        print(f"{batch_duration:.2f}ms")
    return total_duration / BENCHMARK_RUNS_COUNT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()
    results = {}
    for exec_provider in BENCHMARK_EXEC_PROVIDERS:
        session = setup_session(args.model_path, exec_provider)
        speed = benchmark_speed(session, args.batch_size)
        top1_acc, top5_acc = benchmark_accuracy(session, args.data_path, args.batch_size)
        results[exec_provider] = {"top1_acc": top1_acc, "top5_acc": top5_acc, "speed": round(speed, 3)}
    print(results)