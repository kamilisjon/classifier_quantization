import argparse
from pathlib import Path

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from pre_process import load_and_preprocess
from benchmark_onnx import ExecProvider, setup_session, benchmark_accuracy, CalibDataReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, type=Path)
    parser.add_argument("--benchmark_dataset", required=True, type=Path)
    parser.add_argument("--calibrate_dataset", required=True, type=Path)
    args = parser.parse_args()
    output_path = str(args.input_model).rsplit('.', 1)[0] + "_int8.onnx"
    dr = CalibDataReader(args.calibrate_dataset, setup_session(args.input_model, ExecProvider.CPU).get_inputs()[0].name)

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        args.input_model,
        output_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    session = setup_session(output_path, ExecProvider.CUDA)
    print(benchmark_accuracy(session, args.benchmark_dataset, 64))