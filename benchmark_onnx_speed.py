import argparse
import time
from pathlib import Path

import onnxruntime
import numpy as np

def benchmark(model_path: Path):
    session_options = onnxruntime.SessionOptions()
    # 0 = VERBOSE, 1 = INFO, 2 = WARNING, 3 = ERROR, 4 = FATAL
    session_options.log_severity_level = 0  # INFO
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    provider = [('TensorrtExecutionProvider', {
        'trt_fp16_enable': True,
        'trt_int8_enable': True,
        'trt_int8_calibration_table_name': str(model_path.parent / "calibration.flatbuffers"),
        'trt_engine_cache_enable': False
    })]

    # provider = ['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession(str(model_path), sess_options=session_options, providers=provider)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 30

    outputs_meta = session.get_outputs()
    output_names = [o.name for o in outputs_meta]

    input_data = np.zeros((32, 3, 224, 224), np.float32)
    input_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_data, 'cuda', 0)

    iobinding = session.io_binding()
    iobinding.bind_ortvalue_input(input_name, input_ortvalue)
    for name in output_names:
        iobinding.bind_output(name, 'cuda')

    # Warming up
    session.run_with_iobinding(iobinding)
    for i in range(runs):
        start = time.perf_counter()
        session.run_with_iobinding(iobinding)
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    benchmark(args.model_path)