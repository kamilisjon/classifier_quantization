import argparse

import onnx

def extract_conv_weights(model_path: str, quant_model_path: str):
    model = onnx.load(model_path)
    conv_weights = {init.name: onnx.numpy_helper.to_array(init).flatten().tolist() for init in model.graph.initializer}
    print("fc.bias")
    print(conv_weights["fc.bias"])

    quant_model = onnx.load(quant_model_path)
    quant_conv_weights = {init.name: onnx.numpy_helper.to_array(init).flatten().tolist() for init in quant_model.graph.initializer}
    print("fc.bias_quantized")
    print(quant_conv_weights["fc.bias_quantized"])
    
    return conv_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to ONNX model')
    parser.add_argument('quant_model_path', help='Path to ONNX model')
    args = parser.parse_args()
    w = extract_conv_weights(args.model_path, args.quant_model_path)