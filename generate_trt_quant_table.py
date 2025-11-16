import os
from pathlib import Path

import onnxruntime
from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibraterBase, CalibrationDataReader

from pre_process import load_and_preprocess


class CalibDataReader(CalibrationDataReader):
    def __init__(self, folder: Path, input_name: str):
        self.folder, self.input_name, self._iter = folder, input_name, None

    def get_next(self):
        if self._iter is None: self._iter = iter(self.folder.iterdir())
        try: return {self.input_name: load_and_preprocess(next(self._iter))}
        except StopIteration: return None

    def rewind(self): self._iter = None

if __name__ == "__main__":
    model_path = Path("./models/resnet50/resnet50_Opset18_dynamic.onnx")
    calibrate_dataset = Path("C:/Users/kamil/Downloads/calibration_set")
    augmented_model_path = "./augmented_model.onnx"
    calibrator: CalibraterBase = create_calibrator(model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    session = onnxruntime.InferenceSession(model_path, None)
    data_reader = CalibDataReader(calibrate_dataset, session.get_inputs()[0].name)
    calibrator.collect_data(data_reader)
    calibration_data = calibrator.compute_data()
    write_calibration_table(calibration_data, dir=model_path.parent)
    print('Done!')