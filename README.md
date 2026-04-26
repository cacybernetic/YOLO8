
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv --python 3.10
```

```bash
python -m module.train     --config configs/train.yaml      --log-level INFO
python -m module.evaluate  --config configs/eval.yaml       --log-level WARNING
python -m module.infer     --config configs/infer.yaml --image x.jpg --log-level INFO
python -m module.finetuning --config configs/finetune.yaml  --log-level INFO
python -m module.export    --config configs/export.yaml     --log-level DEBUG

python predict.py          --model w.onnx ... --log-level INFO
python live.py             --model w.onnx ... --log-level INFO
```



## Predict Py

```bash
python predict.py --model weights/best.onnx --nc 36 --image samples/image.png --output result.jpg
```
