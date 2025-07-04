import os
import shutil
import argparse

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model = WhisperForConditionalGeneration.from_pretrained(args.hf_model_path)
    model.eval()

    dummy_input_features = torch.randn(1, 80, 3000)
    dummy_input_ids = torch.ones((1, args.decoder_size), dtype=torch.long)

    encoder_path = os.path.join(args.output_dir, "encoder_model.onnx")
    torch.onnx.export(
        model.model.encoder,
        (dummy_input_features,),
        encoder_path,
        input_names=["input_features"],
        output_names=["last_hidden_state"],
        dynamic_axes=None,
        opset_version=17
    )
    encoder_hidden_states = model.model.encoder(dummy_input_features).last_hidden_state
    print("✅ Export Encoder Part @", encoder_path)

    class WhisperDecoderONNX(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.model.decoder
            self.proj_out = model.proj_out

        def forward(self, input_ids, encoder_hidden_states):
            hidden_states = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)[0]
            logits = self.proj_out(hidden_states)
            return logits

    decoder_onnx = WhisperDecoderONNX(model)
    decoder_onnx.eval()

    decoder_path = os.path.join(args.output_dir, "decoder_model.onnx")
    torch.onnx.export(
        decoder_onnx,
        (dummy_input_ids, encoder_hidden_states),
        decoder_path,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=17
    )

    print("✅ Export Decoder Part @", decoder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Runtime Inference')
    parser.add_argument('--hf_model_dir', type=str, help='Huggingface Model Directory')
    parser.add_argument('--output_dir', type=str, help='Onnx Model Directory. Automatically create new one if not existing')
    parser.add_argument('--decoder_size', default=128, type=int, help='Static Size for Decoder Model')
    args = parser.parse_args()
    main(args)
