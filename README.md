
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv --python 3.10
```

```bash
# python -m module.export --config module/export.yaml
python -m export.py --config export.yaml
```

## Predict Py

```bash
python predict.py --model weights/best.onnx --nc 36 --image samples/image.png --output result.jpg
```
