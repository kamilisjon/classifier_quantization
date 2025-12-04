import argparse
import time
from pathlib import Path
from enum import Enum
import csv
import os
import json

import onnxruntime
import numpy as np
from tqdm import tqdm
from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibraterBase, CalibrationDataReader, CalibrationMethod

from pre_process import load_and_preprocess

WARMUP_RUNS_COUNT = 300
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

class CalibDataReader(CalibrationDataReader):
    def __init__(self, folder: Path, input_name: str):
        self.folder, self.input_name, self._iter = folder, input_name, None
    def get_next(self):
        if self._iter is None: self._iter = iter(self.folder.iterdir())
        try: return {self.input_name: load_and_preprocess([next(self._iter)])}
        except StopIteration: return None
    def rewind(self): self._iter = None

def generate_calib_cache(model_path: Path, calib_data_path: Path):
    print('Generating model INT8 calibration table.')
    os.environ["QUANTIZATION_DEBUG"] = "1" 
    data_reader = CalibDataReader(calib_data_path, input_name=setup_session(model_path, ExecProvider.CPU).get_inputs()[0].name)
    calibrator: CalibraterBase = create_calibrator(model_path, [], calibrate_method=CalibrationMethod.MinMax, augmented_model_path=str(model_path).rsplit('.', 1)[0] + "_calib_data_collection.onnx")
    calibrator.collect_data(data_reader)
    calibration_data = calibrator.compute_data()
    write_calibration_table(calibration_data, dir=model_path.parent)
    print('Generated model INT8 calibration table.')

def benchmark_accuracy(session: onnxruntime.InferenceSession, imagenet_data_path: Path, batch_size: int):
    val_dir = imagenet_data_path / 'ILSVRC' / 'Data' / 'CLS-LOC' / "val"
    images = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == '.JPEG'])

    # Parse labels
    labels_csv = imagenet_data_path / "LOC_val_solution.csv"
    imageid_to_label = {}
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            image_id = row["ImageId"]
            pred_str = row["PredictionString"].strip()
            assert pred_str is not None
            tokens = pred_str.split()
            synsets = [tokens[i] for i in range(0, len(tokens), 5)]
            assert len(set(synsets)) == 1  # if there are multiple ground-truth labels, they must be the same
            imageid_to_label[image_id] = synsets[0]
    assert len(imageid_to_label) == len(images)

    # Map synsets
    with open(IMAGENET_LABELS_FILEPATH, "r") as f:
        class_idx = json.load(f)
    gt_label_to_idx = {v[0]: int(k) for k, v in class_idx.items()}

    # Benchmark
    correct_top1 = correct_top5 = 0
    for start in tqdm(range(0, len(images), batch_size), desc="Benchmarking Accuracy"):
        batch_paths = images[start:min(start + batch_size, len(images))]
        outputs = session.run(None, {"x": load_and_preprocess(batch_paths, batch_size)})[0]
        for i in range(len(batch_paths)):
            pred = np.argsort(outputs[i])[-5:][::-1]
            gt_label_idx = gt_label_to_idx[imageid_to_label[batch_paths[i].stem]]

            if pred[0] == gt_label_idx:
                correct_top1 += 1
            if any(p == gt_label_idx for p in pred):
                correct_top5 += 1

    return correct_top1 / len(images) * 100, correct_top5 / len(images) * 100

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
    parser.add_argument("batch_size", type=int)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("benchmark_data_path", type=Path)
    parser.add_argument("calib_data_path", type=Path)
    args = parser.parse_args()
    results = {}
    for exec_provider in [ExecProvider.TRT_INT8, ExecProvider.TRT_FP16, ExecProvider.TRT_FP32, ExecProvider.CUDA]:
        if exec_provider == ExecProvider.TRT_INT8:
            generate_calib_cache(args.model_path, args.calib_data_path)
        session = setup_session(args.model_path, exec_provider)
        speed = benchmark_speed(session, args.batch_size)
        top1_acc, top5_acc = benchmark_accuracy(session, args.benchmark_data_path, args.batch_size)
        results[exec_provider] = {"top1_acc": top1_acc, "top5_acc": top5_acc, "speed": round(speed, 3)}
        print(results)