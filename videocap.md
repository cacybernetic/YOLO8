# YOLOv8 ONNX — Inférence vidéo temps réel

Deux implémentations équivalentes : **Python** (`live.py`) et **Rust** (`yolo_live_rs/`).
Les deux font la même chose : capture vidéo (webcam ou fichier), inférence ONNX,
NMS, et affichage en temps réel avec un rendu futuriste des boites.

---

## Version Python (`live.py`)

### Installation
```bash
pip install numpy onnxruntime opencv-python
```

### Usage
```bash
# Webcam (index 0)
python live.py --model weights/best.onnx --nc 80 --source 0

# Fichier vidéo
python live.py --model weights/best.onnx --nc 80 --source path/to/video.mp4

# Avec sauvegarde du flux annoté
python live.py --model weights/best.onnx --nc 80 --source 0 --output out.mp4

# Mode headless (pas d'affichage, juste sauvegarde)
python live.py --model weights/best.onnx --nc 80 --source video.mp4 \
    --output out.mp4 --no-show
```

Touches : `Q` ou `ESC` pour quitter.

---

## Version Rust (`yolo_live_rs/`)

### Prérequis
- Rust toolchain (rustup)
- Sur Linux : `libv4l-dev` (pour `rustcv` côté caméra) et un compilateur C
- ONNX Runtime sera téléchargé automatiquement à la première compilation
  grâce à la feature `download-binaries` du crate `ort`

### Build
```bash
cd yolo_live_rs
cargo build --release
```

### Usage
```bash
# Webcam
cargo run --release -- --model ../weights/best.onnx --nc 80 --source 0

# Fichier vidéo
cargo run --release -- --model ../weights/best.onnx --nc 80 --source video.mp4

# Mode headless
cargo run --release -- --model ../weights/best.onnx --nc 80 \
    --source video.mp4 --no-show
```

Touches : `Q` (113) ou `ESC` (27) pour quitter.

---

## Mapping Python ↔ Rust (équivalences API)

| Étape                | Python (`live.py`)                  | Rust (`main.rs`)                          |
|----------------------|-------------------------------------|-------------------------------------------|
| Source vidéo         | `cv2.VideoCapture(src)`             | `rustcv::VideoCapture::{new,from_file}`   |
| Lecture frame        | `cap.read()`                        | `cap.read(&mut frame)`                    |
| Letterbox            | `cv2.resize` + numpy padding        | nearest-neighbor manuel sur `Vec<u8>`     |
| Tenseur d'entrée     | `np.ascontiguousarray(...)`          | `ndarray::Array4::<f32>`                  |
| Session ONNX         | `ort.InferenceSession`              | `ort::Session::builder()`                 |
| Inference            | `session.run(...)`                  | `session.run(ort::inputs![...])`          |
| NMS                  | `cv2.dnn.NMSBoxes` (par classe)     | implé. gloutonne en Rust pur              |
| Rendu                | `cv2.line/rectangle/putText`        | manipulation directe des bytes du `Mat`   |
| Affichage            | `cv2.imshow` + `waitKey`            | `rustcv::highgui::imshow` + `wait_key`    |

## Différences notables

1. **Resize** : OpenCV utilise INTER_LINEAR par défaut, la version Rust fait
   du nearest-neighbor pour rester sans dépendance lourde. La qualité est
   acceptable mais peut légèrement dégrader les détections sur petits objets.
   Pour matcher exactement Python, ajouter `fast_image_resize = "5"` au
   `Cargo.toml` et l'utiliser dans `letterbox()`.

2. **Texte** : `rustcv` n'expose pas (encore) `cv::putText`. La version Rust
   omet donc le label texte sur la boite et le HUD FPS. Pour les ajouter,
   utiliser le crate `imageproc::drawing::draw_text_mut` avec une police TTF
   chargée via `rusttype` ou `ab_glyph`.

3. **NMS** : la version Python utilise `cv2.dnn.NMSBoxes` (très rapide, C++).
   La version Rust implémente le NMS gloutonne manuellement pour rester
   indépendante d'OpenCV. Sur ~100 candidats c'est négligeable.

4. **Sauvegarde vidéo** : la version Rust ne le fait pas car `rustcv` n'expose
   pas encore `VideoWriter`. Pour ajouter, utiliser le crate `opencv` (binding
   complet) à la place de `rustcv`, ou écrire frame par frame en PNG/JPEG
   et utiliser `ffmpeg` en post-process.
