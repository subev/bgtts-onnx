# BgTTS-38M V2 as ONNX

An ONNX conversion of the Bulgarian text-to-speech model
[beleata74/BgTTS-38M-V2](https://huggingface.co/beleata74/BgTTS-38M-V2) (revision `3f6ca06b`), with the decode
half of its audio codec, [Aratako/MioCodec-25Hz-24kHz](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz)
(revision `3a737f0d`). It runs on a CPU with ONNX Runtime. The files are attached to the release, not kept in git.

All credit for the model and the codec goes to their authors. This repository only repackages them.

## Files

| File | What it is |
| --- | --- |
| `encoder.onnx` | Text encoder, float32. Inputs `ids` [1, N], `speaker` [1, 128]. Outputs `bias` and the cross-attention keys and values for the eight decoder layers. |
| `decoder.onnx` | One decoder step with an explicit key/value cache, float32. Outputs `logits` [1, 12955], `attention` [3, N] and the new cache. |
| `codec.onnx` | MioCodec's decoder: `codes` [1, T] and `voice` [1, 128] to `real` and `imag` [1, 961, 2T]. Weights stored as float16. |
| `voice-female.bin`, `voice-male.bin`, `voice-male2.bin` | 128 little-endian float32 each: the codec's speaker vector for the three reference recordings published with the model. |

## What was changed

- The model's weights were converted to ONNX without changing their values. The decoder is exported as a single
  step so the sampling loop runs outside the graph.
- The decoder step also returns the cross-attention rows of three heads (layer.head 2.4, 3.2, 5.5). Averaged and
  followed forward through the text, they say which letter each 40 ms frame is speaking.
- Only the codec's decode path is included. The parts that analyse a recording (WavLM and the encoders) are left
  out. Its rotary position code is written with real arithmetic, and the graph ends at the real and imaginary
  spectrum.
- The last step is done outside the graph: an inverse real FFT of 1,920 points per frame, a periodic Hann window,
  overlap-add with a hop of 480, division by the summed squared window, and 720 samples cut from each end. The
  result is 24 kHz audio, 960 samples per code.

Checked against the original PyTorch code: decoder logits within 0.00004 with the same best token on 451 of 451
steps, and the waveform within 0.000006.

## Sampling

As in the model's own recipe: only audio tokens (155 to 12954) and end of speech (4) may be chosen, repetition
penalty 1.1 over the last 100 tokens, temperature 0.3, top-k 250, top-p 0.95. Decoding starts from token 3. Text
is one token per character (see the model's `tokenizer.py`), wrapped in tokens 1 and 2, at most 256 per piece.

## Licences

- BgTTS-38M V2: Apache License 2.0, by beleata74. See `LICENSE-APACHE-2.0.txt`.
- MioCodec-25Hz-24kHz: MIT License, by Aratako. See `LICENSE-MIT-MioCodec.txt`.

The model card does not name the recordings the model was trained on or the speakers of the reference
recordings. Check that this is acceptable for your use.

## SHA-256

```
8ff507c808dc630c90740371ba2e2b7d68ad2457dc386cbb269470fe69c775bc  encoder.onnx
916f90b5347e168efcdffeac6067ca48951d52abef32b29bbf4c4ee515cf09a4  decoder.onnx
2b1ccf8bbf9d1a82f7fe388524ee8bd3f7ab5b60590f07bad21241200d05a26d  codec.onnx
a12a32c78fc0a014b74af7a38096b83584aa5b4f5cef2768c2fd8ac99855ee7a  voice-female.bin
8ed41fd251b89b2ea33970501af0c56c8ce23baf9faf1b7c0897e96439c99bdd  voice-male.bin
cac913059150165b55454f99b6c732f7c7c4b3b3aee86cf6be3562ea2963d288  voice-male2.bin
```
