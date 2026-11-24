#!/usr/bin/env python3
"""Convert pyannote segmentation-3.0 (MIT) to LiteRT with a fixed 10 s input.

The source is the ONNX export shipped with sherpa-onnx (already downloaded for the
speaker spike). Its weights are copied into a plain PyTorch rebuild of the network,
so neither the gated Hugging Face checkpoint nor pyannote.audio is needed.

  segment(x[1,1,160000] f32, 16 kHz audio in [-1,1]) -> y[1,589,7] f32
      log-probabilities over the powerset classes
      {none, s1, s2, s3, s1+s2, s1+s3, s2+s3}; frame shift 270 samples.

Two steps, because the two tool sets live in different environments:

  1. extract  (needs onnx, onnxruntime, soundfile; e.g. build/speaker-spike/venv)
       python scripts/export-pyannote-segmentation-litert.py extract
  2. convert, one of
       --backend litert-torch  (torch + litert_torch; the litert-export venv)
           -> segmentation.tflite, LSTM layers unrolled into ~75k ops
       --backend keras         (tensorflow + tf_keras; build/venvs/original-fp16)
           -> segmentation-fused.tflite, fused sequence-LSTM ops. These keep their
              hidden state between runs: the caller must reset variable tensors
              before every chunk (Interpreter.resetVariableTensors()).

Both steps refuse to continue when a parity check fails.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "build/speaker-spike/models/seg/sherpa-onnx-pyannote-segmentation-3-0"
SOURCE_ONNX = SOURCE_DIR / "model.onnx"
SOURCE_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
OUT_DIR = ROOT / "build/speaker-spike/models/seg/pyannote-segmentation-3-0-litert"
SAMPLES = [
    ROOT / "benchmark-inputs/whisper-suite/reference.wav",
    ROOT / "benchmark-inputs/sample-test/audio.wav",
    ROOT / "benchmark-inputs/article-samples/english-dialogue-short/audio.wav",
    ROOT / "benchmark-inputs/article-samples/english-dialogue-proper-names/audio.wav",
    ROOT / "benchmark-inputs/article-samples/english-dialogue-mixed-fragments/audio.wav",
]
SR, WINDOW, FRAMES, CLASSES, HIDDEN = 16000, 160000, 589, 7, 128
TORCH_MAX_ABS = 1e-3        # PyTorch rebuild vs ONNX Runtime
LITERT_MAX_ABS = 1e-2       # LiteRT vs ONNX Runtime
LITERT_MIN_ARGMAX = 0.999


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_chunks(per_file=4, hop_s=5.0):
    """A few real 10 s chunks per benchmark sample, zero-padded at the end."""
    import soundfile as sf
    chunks = []
    for path in SAMPLES:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        assert rate == SR and audio.shape[1] == 1, f"{path}: expected 16 kHz mono"
        audio = audio[:, 0]
        for index in range(per_file):
            start = int(index * hop_s * SR)
            if start >= len(audio):
                break
            chunk = np.zeros(WINDOW, np.float32)
            piece = audio[start:start + WINDOW]
            chunk[:len(piece)] = piece
            chunks.append(chunk)
    return np.stack(chunks)


def compare(name, got, want, max_abs_limit, min_argmax):
    max_abs = float(np.abs(got - want).max())
    mean_abs = float(np.abs(got - want).mean())
    argmax = float((got.argmax(-1) == want.argmax(-1)).mean())
    ok = max_abs <= max_abs_limit and argmax >= min_argmax
    print(f"{name}: max abs {max_abs:.2e} (limit {max_abs_limit:.0e}), mean abs {mean_abs:.2e}, "
          f"same arg-max {argmax:.5f} (min {min_argmax}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"max_abs": max_abs, "mean_abs": mean_abs, "argmax_agreement": argmax, "passed": ok}


# ------------------------------------------------------------------ step 1: extract
def extract(args):
    import onnx
    import onnx.numpy_helper as nh
    import onnxruntime as ort

    actual = sha256(SOURCE_ONNX)
    if actual != SOURCE_SHA256:
        sys.exit(f"Unexpected source model hash {actual}")
    init = {i.name: nh.to_array(i) for i in onnx.load(str(SOURCE_ONNX)).graph.initializer}

    def reorder(block):
        """ONNX gate order i,o,f,c -> PyTorch order i,f,g(c),o along axis 0."""
        i, o, f, c = np.split(block, 4, axis=0)
        return np.concatenate([i, f, c, o], axis=0)

    weights = {
        "wav_norm.weight": init["ortshared_1_1_1_1_token_110"],
        "wav_norm.bias": init["ortshared_1_1_1_0_token_107"],
        "sinc_filters": init["/sincnet/conv1d.0/Concat_2_output_0"],
        "linear0.weight": init["onnx::MatMul_915"].T, "linear0.bias": init["linear.0.bias"],
        "linear1.weight": init["onnx::MatMul_916"].T, "linear1.bias": init["linear.1.bias"],
        "classifier.weight": init["onnx::MatMul_917"].T, "classifier.bias": init["ortshared_1_1_7_0_token_109"],
    }
    for k in (0, 1, 2):
        weights[f"norm{k}.weight"] = init[f"sincnet.norm1d.{k}.weight"]
        weights[f"norm{k}.bias"] = init[f"sincnet.norm1d.{k}.bias"]
    for k in (1, 2):
        weights[f"conv{k}.weight"] = init[f"sincnet.conv1d.{k}.weight"]
        weights[f"conv{k}.bias"] = init[f"sincnet.conv1d.{k}.bias"]
    for layer, (b, w, r) in enumerate([(783, 784, 785), (826, 827, 828), (869, 870, 871), (912, 913, 914)]):
        B, W, R = init[f"onnx::LSTM_{b}"], init[f"onnx::LSTM_{w}"], init[f"onnx::LSTM_{r}"]
        for direction, suffix in enumerate(("", "_reverse")):
            weights[f"lstm.weight_ih_l{layer}{suffix}"] = reorder(W[direction])
            weights[f"lstm.weight_hh_l{layer}{suffix}"] = reorder(R[direction])
            weights[f"lstm.bias_ih_l{layer}{suffix}"] = reorder(B[direction][:4 * HIDDEN])
            weights[f"lstm.bias_hh_l{layer}{suffix}"] = reorder(B[direction][4 * HIDDEN:])
    assert weights["sinc_filters"].shape == (80, 1, 251) and weights["classifier.weight"].shape == (CLASSES, HIDDEN)

    x = reference_chunks()
    session = ort.InferenceSession(str(SOURCE_ONNX), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    y = np.concatenate([session.run(None, {"x": chunk[None, None, :]})[0] for chunk in x])
    onnx_ms = 1000 * (time.perf_counter() - started) / len(x)
    assert y.shape == (len(x), FRAMES, CLASSES), y.shape

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "weights.npz", **{k: np.ascontiguousarray(v, np.float32) for k, v in weights.items()})
    np.savez(args.out / "reference.npz", x=x, y=y.astype(np.float32), onnx_ms_per_chunk=onnx_ms)
    print(f"extracted {len(weights)} tensors and {len(x)} reference chunks to {args.out} "
          f"(ONNX Runtime {onnx_ms:.1f} ms per chunk on this host)")


# ------------------------------------------------------------------ step 2: convert
def build_model(weights):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class PyanNetFixed(nn.Module):
        """SincNet -> 4-layer bidirectional LSTM -> 2 linear -> classifier, fixed 10 s input."""

        def __init__(self):
            super().__init__()
            self.wav_norm = nn.InstanceNorm1d(1, affine=True)
            self.register_buffer("sinc_filters", torch.zeros(80, 1, 251))
            self.norm0 = nn.InstanceNorm1d(80, affine=True)
            self.conv1 = nn.Conv1d(80, 60, 5)
            self.norm1 = nn.InstanceNorm1d(60, affine=True)
            self.conv2 = nn.Conv1d(60, 60, 5)
            self.norm2 = nn.InstanceNorm1d(60, affine=True)
            self.lstm = nn.LSTM(60, HIDDEN, num_layers=4, bidirectional=True, batch_first=True)
            self.linear0 = nn.Linear(2 * HIDDEN, HIDDEN)
            self.linear1 = nn.Linear(HIDDEN, HIDDEN)
            self.classifier = nn.Linear(HIDDEN, CLASSES)

        def forward(self, x):
            x = self.wav_norm(x)
            x = torch.abs(F.conv1d(x, self.sinc_filters, stride=10))
            x = F.leaky_relu(self.norm0(F.max_pool1d(x, 3)))
            x = F.leaky_relu(self.norm1(F.max_pool1d(self.conv1(x), 3)))
            x = F.leaky_relu(self.norm2(F.max_pool1d(self.conv2(x), 3)))
            x, _ = self.lstm(x.transpose(1, 2))
            x = F.leaky_relu(self.linear0(x))
            x = F.leaky_relu(self.linear1(x))
            return F.log_softmax(self.classifier(x), dim=-1)

    model = PyanNetFixed()
    state = {name: torch.from_numpy(np.array(value)) for name, value in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return model.eval()


def run_litert(path, x, threads=4):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:                              # TensorFlow venv
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(path), num_threads=threads)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    results, started = [], None
    for index, chunk in enumerate(x):
        if index == 1:
            started = time.perf_counter()        # first run excluded from timing
        interpreter.reset_all_variables()        # fused LSTM ops keep their state between runs
        interpreter.set_tensor(inp["index"], chunk[None, None, :])
        interpreter.invoke()
        results.append(interpreter.get_tensor(out["index"]).copy())
    ms = 1000 * (time.perf_counter() - started) / max(1, len(x) - 1)
    ops = sorted({d["op_name"] for d in interpreter._get_ops_details()})
    return np.concatenate(results), ms, ops, len(interpreter._get_ops_details())


def build_keras(weights):
    """Same network in Keras so the converter emits fused sequence-LSTM ops."""
    import tensorflow as tf
    import tf_keras as keras
    L = keras.layers

    def instance_norm(x, name):
        weight, bias = weights[f"{name}.weight"], weights[f"{name}.bias"]
        def fn(v):
            mean = tf.reduce_mean(v, axis=1, keepdims=True)
            var = tf.reduce_mean(tf.square(v - mean), axis=1, keepdims=True)
            return (v - mean) * tf.math.rsqrt(var + 1e-5) * weight + bias
        return L.Lambda(fn, name=name)(x)

    inp = keras.Input(batch_shape=(1, 1, WINDOW), name="x")
    x = L.Reshape((WINDOW, 1))(inp)
    x = instance_norm(x, "wav_norm")
    sinc = L.Conv1D(80, 251, strides=10, use_bias=False, name="sinc")
    x = L.Lambda(tf.abs)(sinc(x))
    x = L.LeakyReLU(0.01)(instance_norm(L.MaxPool1D(3)(x), "norm0"))
    conv1 = L.Conv1D(60, 5, name="conv1"); x = L.LeakyReLU(0.01)(instance_norm(L.MaxPool1D(3)(conv1(x)), "norm1"))
    conv2 = L.Conv1D(60, 5, name="conv2"); x = L.LeakyReLU(0.01)(instance_norm(L.MaxPool1D(3)(conv2(x)), "norm2"))
    lstms = []
    for layer in range(4):
        lstms.append(L.Bidirectional(L.LSTM(HIDDEN, return_sequences=True), merge_mode="concat", name=f"lstm{layer}"))
        x = lstms[-1](x)
    dense0 = L.Dense(HIDDEN, name="linear0"); x = L.LeakyReLU(0.01)(dense0(x))
    dense1 = L.Dense(HIDDEN, name="linear1"); x = L.LeakyReLU(0.01)(dense1(x))
    classifier = L.Dense(CLASSES, name="classifier")
    out = L.Lambda(lambda v: tf.nn.log_softmax(v, axis=-1), name="y")(classifier(x))
    model = keras.Model(inp, out)

    sinc.set_weights([weights["sinc_filters"].transpose(2, 1, 0)])
    conv1.set_weights([weights["conv1.weight"].transpose(2, 1, 0), weights["conv1.bias"]])
    conv2.set_weights([weights["conv2.weight"].transpose(2, 1, 0), weights["conv2.bias"]])
    for layer, bidi in enumerate(lstms):
        values = []
        for suffix in ("", "_reverse"):            # PyTorch and Keras share gate order i,f,c,o
            values += [weights[f"lstm.weight_ih_l{layer}{suffix}"].T, weights[f"lstm.weight_hh_l{layer}{suffix}"].T,
                       weights[f"lstm.bias_ih_l{layer}{suffix}"] + weights[f"lstm.bias_hh_l{layer}{suffix}"]]
        bidi.set_weights(values)
    for dense, name in ((dense0, "linear0"), (dense1, "linear1"), (classifier, "classifier")):
        dense.set_weights([weights[f"{name}.weight"].T, weights[f"{name}.bias"]])
    return model


def convert(args):
    weights = dict(np.load(args.out / "weights.npz"))
    reference = np.load(args.out / "reference.npz")
    x, y = reference["x"], reference["y"]
    started = time.perf_counter()
    if args.backend == "litert-torch":
        import torch
        model = build_model(weights)
        with torch.no_grad():
            rebuilt = np.concatenate([model(torch.from_numpy(c)[None, None, :]).numpy() for c in x])
    else:
        import tensorflow as tf
        model = build_keras(weights)
        rebuilt = np.concatenate([model(c[None, None, :], training=False).numpy() for c in x])
    gate1 = compare(f"{args.backend} rebuild vs ONNX Runtime", rebuilt, y, TORCH_MAX_ABS, 1.0)
    if not gate1["passed"]:
        sys.exit("Rebuild parity failed; not exporting.")
    if args.check_only:
        return

    stem = "segmentation" if args.backend == "litert-torch" else "segmentation-fused"
    path = args.out / f"{stem}.tflite"
    started = time.perf_counter()
    if args.backend == "litert-torch":
        import litert_torch
        litert_torch.signature("segment", model, (torch.zeros(1, 1, WINDOW),)).convert().export(str(path))
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        path.write_bytes(converter.convert())
        # The Keras->LiteRT LSTM fusion hard-codes cell_clip=10, which this model exceeds
        # (0.5 max abs error, 99.6% arg-max agreement). PyTorch applies no clipping.
        from tensorflow.lite.tools import flatbuffer_utils
        flat = flatbuffer_utils.read_model(str(path))
        patched = 0
        for subgraph in flat.subgraphs:
            for op in subgraph.operators:
                if type(op.builtinOptions).__name__ == "UnidirectionalSequenceLSTMOptionsT":
                    op.builtinOptions.cellClip = 0.0
                    patched += 1
        assert patched == 8, patched
        flatbuffer_utils.write_model(flat, str(path))
    print(f"exported {path} ({path.stat().st_size} bytes) in {time.perf_counter() - started:.0f} s", flush=True)

    got, litert_ms, ops, op_count = run_litert(path, x)
    gate2 = compare("LiteRT vs ONNX Runtime", got, y, LITERT_MAX_ABS, LITERT_MIN_ARGMAX)
    onnx_ms = float(reference["onnx_ms_per_chunk"])
    print(f"host speed: LiteRT {litert_ms:.1f} ms per chunk, ONNX Runtime {onnx_ms:.1f} ms; {op_count} ops: {', '.join(ops)}")
    sidecar = {
        "model": "pyannote segmentation-3.0", "license": "MIT (see LICENSE)",
        "source_onnx_sha256": SOURCE_SHA256, "backend": args.backend, "quantisation": "none",
        "requires_state_reset_between_runs": args.backend == "keras",
        "input": [1, 1, WINDOW], "output": [1, FRAMES, CLASSES],
        "sample_rate": SR, "frame_shift_samples": 270, "receptive_field_samples": 991,
        "bytes": path.stat().st_size, "sha256": sha256(path), "op_count": op_count, "ops": ops,
        "parity_rebuild_vs_onnx": gate1, "parity_litert_vs_onnx": gate2,
        "host_ms_per_chunk": {"litert": round(litert_ms, 1), "onnxruntime": round(onnx_ms, 1)},
        "reference_chunks": int(len(x)),
    }
    (args.out / f"{stem}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    shutil.copyfile(SOURCE_DIR / "LICENSE", args.out / "LICENSE")
    if not gate2["passed"]:
        sys.exit("LiteRT parity failed; do not use this file.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["extract", "convert"])
    parser.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--backend", choices=["litert-torch", "keras"], default="litert-torch",
                        help="convert: litert-torch unrolls the LSTM layers; keras (TensorFlow venv) keeps them fused")
    parser.add_argument("--check-only", action="store_true", help="convert: run the rebuild parity check and stop")
    args = parser.parse_args()
    (extract if args.step == "extract" else convert)(args)


if __name__ == "__main__":
    main()
