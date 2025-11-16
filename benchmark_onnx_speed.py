import argparse
import time
from pathlib import Path
from enum import Enum

import onnxruntime
import numpy as np


WARMUP_RUNS_COUNT = 25
BENCHMARK_RUNS_COUNT = 50

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
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()
    results = {}
    for exec_provider in BENCHMARK_EXEC_PROVIDERS:
        session = setup_session(args.model_path, exec_provider)
        speed = benchmark_speed(session, args.batch_size)
        results[exec_provider] = speed
    print(results)