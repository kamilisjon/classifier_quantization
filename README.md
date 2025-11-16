Purpose of this experiment is to explore INT8 quantization of classification models.
This was done by following [ONNX Runtime instructions](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md).

## Environment setup instructions ##
* Pull this repository
* `cd ./classifier_quantization`
* `conda create -n classifier_quantization python=3.12`
* `conda activate classifier_quantization`
* `pip install -r requirements.txt`

## How to use this? ##
* [Download and unzip the ImageNet dataset](https://www.image-net.org/download.php).
    * Will need train set for INT8 quantization calibration set amd validation set for benchmarking.
* Benchmark model on ImageNet validation set.
    * `python .\benchmark_onnx.py "path-to-model" "path-to-imagenet-dataset"`
    * [Here](https://github.com/onnx/models/tree/main/Computer_Vision) you can find many models to experiment with quantization. To see how resilient different models are to quantization. I was using a timm [ResNet18](https://github.com/onnx/models/tree/main/Computer_Vision/resnet18_Opset18_timm) model.