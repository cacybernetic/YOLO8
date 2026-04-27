
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv --python 3.10
```

```bash
python -m yolov8.train     --config configs/train.yaml      --log-level INFO
python -m yolov8.evaluate  --config configs/eval.yaml       --log-level INFO
python -m yolov8.infer     --config configs/infer.yaml --image x.jpg --log-level INFO
python -m yolov8.finetuning --config configs/finetuning.yaml  --log-level INFO
python -m yolov8.export    --config configs/export.yaml     --log-level DEBUG

python predict.py          --model w.onnx ... --log-level INFO
python live.py             --model w.onnx ... --log-level INFO
```


## Predict Py

```bash
python predict.py --model weights/best.onnx --nc 36 --image samples/image.png --output result.jpg
```
