import os

from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibraterBase

from imagenet_calibration_datareader import ResNet50DataReader


if __name__ == "__main__":
    model_path = "./resnet18_Opset18_dynamic.onnx"
    calibrate_dataset = "C:/Users/kamil/Downloads/calibration_set"
    augmented_model_path = "./augmented_model.onnx"
    batch_size = 20
    calibration_dataset_size = 1000  # Size of dataset for calibration

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    execution_provider = ["TensorrtExecutionProvider"]

    # Generate INT8 calibration table
    calibrator: CalibraterBase = create_calibrator(model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    data_reader = ResNet50DataReader(calibrate_dataset, model_path)
    calibrator.collect_data(data_reader)
    calibration_data = calibrator.compute_data()
    print(calibration_data)
    write_calibration_table(calibration_data)