import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, CalibrationDataReader

from pre_process import load_and_preprocess
from benchmark_onnx import ExecProvider, setup_session, benchmark_accuracy


def _preprocess_images(images_folder: Path, size_limit=0):
    image_names = os.listdir(str(images_folder))
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        unconcatenated_batch_data.append(load_and_preprocess([images_folder / image_name]))
    return np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, type=Path)
    parser.add_argument("--benchmark_dataset", required=True, type=Path)
    parser.add_argument("--calibrate_dataset", required=True, type=Path)
    args = parser.parse_args()
    output_path = str(args.input_model).rsplit('.', 1)[0] + "_int8.onnx"
    dr = ResNet50DataReader(args.calibrate_dataset, str(args.input_model))

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