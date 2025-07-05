# Whisper
### Download wisper model from huggingface hub (whisper-small as example)
```
git clone https://huggingface.co/openai/whisper-small
```
### Convert HF model to ONNX model
```
python export_onnx.py --hf_model_dir <hf_model_dir> --output_dir whisper-small --decoder_size 128
```
### Quantization (Optional)
- Encoder Part
```
python quark_quantize.py --model_path whisper-small\encoder_model.onnx --output_dir whisper-small-bf16
```
- Decoder Part
```
python quark_quantize.py --model_path whisper-small\decoder_model.onnx --output_dir whisper-small-bf16
```
### Run
```
python run.py --hf_model_dir <hf_model_dir> --onnx_model_dir whisper-small --decoder_size 128
python run.py --hf_model_dir <hf_model_dir> --onnx_model_dir whisper-small-bf16 --decoder_size 128
```
### Benchmark
| Precision | Source Time (s) | Encoding Time (s) | Decoding Time (s) |
|-----------|------------------|-------------------|-------------------|
| FP32      | 11.0             | 1.1726            | 10.3123           |

