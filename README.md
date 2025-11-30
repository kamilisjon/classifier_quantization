Purpose of this experiment is to explore INT8 quantization of classification models.
This was done by following [ONNX Runtime instructions](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md).

## Environment setup instructions ##
* Pull this repository
* `cd ./classifier_quantization`
* `conda create -n classifier_quantization python=3.12`
* `conda activate classifier_quantization`
* `pip install -r requirements.txt`
* CUDA and TensorRT need to be installed in the global environment.

## Quantize and compare models speed and accuracy ##
* Prepare data (this step will need to be done only once):
    * [Download and unzip the ImageNet dataset](https://www.image-net.org/download.php).
        * Will need train set for INT8 quantization calibration set and validation set for benchmarking.
    * Generate calibration dataset:
        * `python .\sample_calibration_set.py "path-to-imagenet-train-dataset" "path-to-save-calibration-dataset" --num-samples 10`
* Prepare model (this step will need to be done only once per model):
    * This repository assumes timm style preprocessing.
    * [Here](https://github.com/onnx/models/tree/main/Computer_Vision) you can find many timm style ONNX models to experiment with quantization. For example, resnet18 model can found [here](https://github.com/onnx/models/tree/main/Computer_Vision/resnet18_Opset18_timm).
    * If model batch size is not dynamic, you will need to change it to dynamic.
        * `python .\make_batch_size_dynamic.py "path-to-model"`
* Quantize model and benchmark on ImageNet validation set. Also fp32, fp16 benchmarks will be done for comparison using TensorRT. 
    * `python .\benchmark_onnx.py 64 "path-to-model" "path-to-imagenet-dataset" "path-to-calibration-dataset"`

## Compare FP32 and INT8 models weights matrces ##
* For weights comparison, we need to prepare INT8 quantized ONNX model first.
    * `python .\quantize_onnx.py "path-to-model" "path-to-imagenet-dataset" "path-to-calibration-dataset"`
* `python .\compare_models.py "path-to-fp32-model" "path-to-int8-model"`