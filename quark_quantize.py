import os
import argparse

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config


def main(args):
    # Use default quantization configuration
    quant_config = get_default_config("BF16")
    quant_config.extra_options["BF16QDQToCast"] = True
    config = Config(global_quant_config=quant_config)
    config.global_quant_config.extra_options["UseRandomData"] = True
    
    config = Config(global_quant_config=quant_config)
    config.global_quant_config.extra_options["UseRandomData"] = True
    print("The configuration of the quantization is {}".format(config))

    # Create an ONNX Quantizer
    quantizer = ModelQuantizer(config)

    input_model_path = os.path.join(args.model_path)
    output_model_path = os.path.join(args.output_dir, os.path.basename(input_model_path))

    os.makedirs(args.output_dir,exist_ok=True)
    
    # Quantize the ONNX model
    quant_model = quantizer.quantize_model(model_input = input_model_path,
                                       model_output = output_model_path,
                                       calibration_data_path = None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, help='Onnx Model Path')
    parser.add_argument('--output_dir', type=str, help='Output directory for the ONNX model')

    args = parser.parse_args()

    main(args)