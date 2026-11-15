# BgTTS-38M V2 as ONNX

An ONNX conversion of the Bulgarian text-to-speech model
[beleata74/BgTTS-38M-V2](https://huggingface.co/beleata74/BgTTS-38M-V2) (revision `3f6ca06b`), with the decode
half of its audio codec, [Aratako/MioCodec-25Hz-24kHz](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz)
(revision `3a737f0d`). It runs on a CPU with ONNX Runtime. The files are attached to the release, not kept in git.

All credit for the model and the codec goes to their authors. This repository only repackages them.

## Files

Release `2026-11-15.2` (use this one). Each graph is a small structure file and a side file of float32 weights;
keep the two together. ONNX Runtime maps a side file from disk instead of copying it, so the three graphs hold
about 200 MB of memory instead of about 830 MB.

| File | What it is |
| --- | --- |
| `encoder.onnx` + `encoder.data` | Text encoder. Inputs `ids` [1, N], `speaker` [1, 128]. Outputs `bias` and the cross-attention keys and values for the eight decoder layers. |
| `decoder.onnx` + `decoder.data` | One decoder step with an explicit key/value cache. Outputs `logits` [1, 12955], `attention` [3, N] and the new cache. |
| `codec.onnx` + `codec.data` | MioCodec's decoder: `codes` [1, T] and `voice` [1, 128] to `real` and `imag` [1, 961, 2T]. |
| `voice-female.bin`, `voice-male.bin`, `voice-male2.bin` | 128 little-endian float32 each: the codec's speaker vector for the three reference recordings published with the model. |

To keep memory low, set the session option `session.disable_prepacking` to `1` for the encoder and the codec. Leave
it off for the decoder, which runs 25 times per second of speech and is about twice as fast with prepacking.

Release `2026-11-15` holds the first packaging: weights inside each graph file, the codec at float16. It sounds
the same and uses about four times the memory.

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

## SHA-256 (release 2026-11-15.2)

```
a3a57f9541e073ccc40ff336db98a6b265c9207fee15c6f8d56546f5fc3ed5b2  encoder.onnx
2837dfb9c6d945e0e73c3623e762faccac74085f6ee9eba3501801044c788027  encoder.data
a6d9d87e7fa2629daf77c1b56b4dd864a43d3f4461ca4752a2bd312262a90d0f  decoder.onnx
12f693bd8597e86b1e6f2efe080f2933554d1d6cf46f532e77ace3767d235c14  decoder.data
fd2ce4d3ae355cbb831701cd3b203d60805577744c7d83f15fb785c59ebdeb9e  codec.onnx
45f5e4dd97f1e33e6877627627205193945231976591ec82c94efae0b5bbeab7  codec.data
a12a32c78fc0a014b74af7a38096b83584aa5b4f5cef2768c2fd8ac99855ee7a  voice-female.bin
8ed41fd251b89b2ea33970501af0c56c8ce23baf9faf1b7c0897e96439c99bdd  voice-male.bin
cac913059150165b55454f99b6c732f7c7c4b3b3aee86cf6be3562ea2963d288  voice-male2.bin
```
