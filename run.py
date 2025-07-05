import os
import argparse
import time

import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import librosa
import numpy as np



def main(args):
   tok = WhisperTokenizer.from_pretrained(args.hf_model_dir)

   # Load and Compile Model
   encoder_model_path = os.path.join(args.onnx_model_dir, "encoder_model.onnx")
   encoder_cache_dir = os.path.join(os.getcwd(), os.path.basename(args.onnx_model_dir), "encoder_model")
   os.makedirs(encoder_cache_dir, exist_ok=True)
   decoder_model_path = os.path.join(args.onnx_model_dir, "decoder_model.onnx")
   decoder_cache_dir = os.path.join(os.getcwd(), os.path.basename(args.onnx_model_dir), "decoder_model")
   os.makedirs(decoder_cache_dir, exist_ok=True)

   enc_sess = ort.InferenceSession(
        encoder_model_path,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": "vaiml_config.json",
                           "cache_dir": encoder_cache_dir,
                           "cacheKey": "modelcachekey",
                           }],
    )

   dec_sess = ort.InferenceSession(
      decoder_model_path,
      providers=["VitisAIExecutionProvider"],
      provider_options=[{"config_file": "vaiml_config.json",
                        "cache_dir": decoder_cache_dir,
                        "cacheKey": "modelcachekey",
                        }],
   )

   if args.vaiml_compile:
      return
   
   tok = WhisperTokenizer.from_pretrained(args.hf_model_dir)

   # Input Data
   audio, sr = librosa.load(args.sample_path, sr=16000)
   feat = WhisperFeatureExtractor.from_pretrained(args.hf_model_dir)
   inputs = feat(audio, return_tensors="np", sampling_rate=16000)
   input_features = inputs["input_features"]
   

   start_enc = time.time()
   encoder_outputs = enc_sess.run(
      ["last_hidden_state"],
      {"input_features": input_features}
   )[0]
   end_enc = time.time()

   pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

   decoder_input_ids = np.array([[tok.convert_tokens_to_ids("<|startoftranscript|>")]], dtype=np.int64)
   generated = []
   start_dec = time.time()
   while True:
      cur_len = decoder_input_ids.shape[1]
      if cur_len < args.decoder_size:
         padding = np.full((1, args.decoder_size - cur_len), pad_token_id, dtype=np.int64)
         model_input_ids = np.concatenate([decoder_input_ids, padding], axis=1)
      else:
         model_input_ids = decoder_input_ids[:, :args.decoder_size]

      logits = dec_sess.run(
         ["logits"],
         {"input_ids": model_input_ids, "encoder_hidden_states": encoder_outputs}
      )[0]

      next_id = np.argmax(logits[:, cur_len-1, :], axis=-1)
      token = next_id.item()
      if token == tok.eos_token_id:
         break
      decoder_input_ids = np.concatenate([decoder_input_ids, next_id[:, None]], axis=-1)
      # print(tok.decode(next_id.item()), end='', flush=True)
      generated.append(token)
   # print()
   end_dec = time.time()

   print(tok.decode(generated))
   print(f"Source time {audio.shape[0]/sr}")
   print(f"Encoding time: {end_enc-start_enc}")
   print(f"Decoding time: {end_dec-start_dec}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Runtime Inference')
    parser.add_argument('--hf_model_dir', type=str, help='Huggingface Model Directory')
    parser.add_argument('--onnx_model_dir', type=str, help='Onnx Model Directory')
    parser.add_argument('--vaiml_compile', action='store_true', help='Compile Model on Linux')
    parser.add_argument('--decoder_size', default=128, type=int, help='Static Size for Decoder Model')
    parser.add_argument('--sample_path', default="samples/jfk.wav", type=str, help='Voice sample for recognization')
    args = parser.parse_args()
    main(args)
