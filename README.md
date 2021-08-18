# csam_collision

1. Use https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX to obtain the model and convert it to onnx.
2. Use `onnx_tf` to convert the onnx model to tensorflow and save it in `model.pb`.
3. Run the script. It will start off making lots of progress and then slowly approach a loss of -1. If you don't succeed, try raising the `eps` value in the script and play around with `eps_step` to control the noisyness.
4. Confirm using `nnhash.py` from https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX.

Note: The final output might have a slightly different hash due to quantization and clipping. Just try again until you find an exact match.

Images from wikipedia

Concurrent work: https://github.com/anishathalye/neural-hash-collider

Pull requests to improve this repo welcome.
