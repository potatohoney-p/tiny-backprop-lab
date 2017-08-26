# Tiny Backprop Lab

A small sigmoid feed-forward neural network with explicit gradients and a deterministic XOR demonstration.

## Run locally

Use a currently supported Python 3 interpreter. No third-party packages, credentials, downloads, or network access are required.

```sh
python -m unittest discover -s tests -v
python demo.py
```

Run from the repository directory. The demo reads `examples/input.json`; inputs are reproducible, authored examples.

## Scope and behavior

A two-layer network uses sigmoid activations, squared-error loss and online gradient descent. Every derivative is implemented directly and checked against central finite differences. XOR inputs are generated mathematical examples. This is a teaching implementation, not a tensor framework, and does not use TensorFlow or a GPU.

## Layout

- `core.py`: primary algorithm or storage contract.
- `features.py`: pipeline or integration layer.
- `tests/`: independent behavioral assertions, failure cases, and regressions.
- `demo.py` and `examples/input.json`: repeatable end-to-end example.

## Example output

```json
{
  "gradient_error": 1.011614403356731e-12,
  "predictions": [
    0.0077,
    0.985,
    0.985,
    0.0222
  ]
}
```

## Compatibility and license

The implementation uses language features and standard-library APIs available in [Python 3.4](https://docs.python.org/3.4/library/index.html), released in [March 2014](https://www.python.org/downloads/release/python-340/). Use a supported runtime; an old interpreter is not required. Compatibility with a historical runtime is not certified by current-runtime tests.

MIT licensed. No third-party source code or datasets are bundled.
