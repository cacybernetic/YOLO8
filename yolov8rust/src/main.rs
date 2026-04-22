//! # YOLOv8 ONNX — Inférence complète en Rust
//!
//! Pré-traitement (letterbox + normalisation), inférence ONNX Runtime,
//! post-traitement (décodage cxcywh → xyxy, filtre confiance, NMS par classe),
//! reprojection dans l'espace image d'origine, rendu des boites et sauvegarde.
//!
//! ## Cargo.toml
//! ```toml
//! [package]
//! name    = "yolov8_infer"
//! version = "0.1.0"
//! edition = "2021"
//!
//! [dependencies]
//! anyhow  = "1"
//! clap    = { version = "4", features = ["derive"] }
//! image   = "0.25"
//! ndarray = "0.16"
//! ort     = { version = "2", features = ["download-binaries"] }
//! ```
//!
//! ## Usage
//! ```bash
//! cargo run --release -- \
//!     --model  weights/best.onnx \
//!     --image  photo.jpg        \
//!     --output result.jpg       \
//!     --conf   0.25             \
//!     --iou    0.45             \
//!     --size   640              \
//!     --nc     80
//! ```
//!
//! Le modèle ONNX doit être exporté avec `export.py` (wrapper `YoloExportWrapper`)
//! dont la sortie a la forme  [B, 4+nc, num_anchors] :
//!   - lignes 0..4   : (cx, cy, w, h) en pixels de l'image letterboxée
//!   - lignes 4..4+nc: scores de classe déjà sigmoïdés

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::{bail, Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{Array4, ArrayViewD};
use ort::{session::Session, value::Tensor as OrtTensor};
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────────
// Noms de classes COCO (80 classes, ordre standard)
// ─────────────────────────────────────────────────────────────────────────────

const COCO_CLASSES: &[&str] = &[
    "person",        "bicycle",      "car",           "motorcycle",   "airplane",
    "bus",           "train",        "truck",         "boat",         "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",         "sheep",        "cow",
    "elephant",      "bear",         "zebra",         "giraffe",      "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",         "baseball bat",
    "baseball glove","skateboard",   "surfboard",     "tennis racket","bottle",
    "wine glass",    "cup",          "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",         "sandwich",     "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",        "donut",
    "cake",          "chair",        "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",            "laptop",       "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",    "oven",
    "toaster",       "sink",         "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",   "toothbrush",
];

/// Palette de couleurs RGB (20 couleurs distinctes, cyclée par class_id)
const PALETTE: [[u8; 3]; 20] = [
    [255,  56,  56], [255, 157, 151], [255, 112,  31], [255, 178,  29],
    [207, 210,  49], [ 72, 249,  10], [146, 204,  23], [ 61, 219, 134],
    [ 26, 147,  52], [  0, 212, 187], [ 44, 153, 168], [  0, 194, 255],
    [ 52,  69, 147], [100, 115, 255], [  0,  24, 236], [132,  56, 255],
    [ 82,   0, 133], [203,  56, 255], [255, 149, 200], [255,  55, 199],
];

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "yolov8_infer", about = "Inférence YOLOv8 ONNX en Rust")]
struct Args {
    /// Chemin vers le fichier .onnx
    #[arg(short, long)]
    model: PathBuf,

    /// Chemin vers l'image d'entrée (JPEG / PNG / …)
    #[arg(short, long)]
    image: PathBuf,

    /// Chemin de l'image annotée en sortie (optionnel)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Seuil de confiance [0, 1]
    #[arg(long, default_value_t = 0.25)]
    conf: f32,

    /// Seuil IoU pour le NMS [0, 1]
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// Taille d'entrée du modèle (carré, ex : 640)
    #[arg(long, default_value_t = 640)]
    size: u32,

    /// Nombre de classes du modèle
    #[arg(long, default_value_t = 80)]
    nc: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Structures de données
// ─────────────────────────────────────────────────────────────────────────────

/// Une boite de détection (coordonnées dans l'espace image d'origine)
#[derive(Debug, Clone)]
struct Detection {
    /// Coins supérieur-gauche et inférieur-droit en pixels
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    /// Score de confiance (probabilité max de classe)
    confidence: f32,
    /// Indice de la classe prédite
    class_id: usize,
}

/// Résultat d'un letterbox (image carrée + méta-données de reprojection)
struct LetterboxResult {
    /// Image redimensionnée + paddée, format RGB
    image: RgbImage,
    /// Facteur d'échelle appliqué (unique — aspect ratio conservé)
    ratio: f32,
    /// (pad_gauche, pad_haut) en pixels
    pad: (u32, u32),
}

// ─────────────────────────────────────────────────────────────────────────────
// Pré-traitement
// ─────────────────────────────────────────────────────────────────────────────

/// Redimensionne l'image en conservant le ratio, puis la centre sur un canvas
/// `target_size × target_size` rempli de gris (114, 114, 114).
///
/// Reproduit fidèlement la fonction `letterbox` de `dataset.py`.
fn letterbox(img: &DynamicImage, target_size: u32) -> LetterboxResult {
    let (orig_w, orig_h) = img.dimensions();

    // Ratio unique pour les deux axes (on prend le plus petit pour tout faire
    // rentrer sans débordement)
    let ratio = (target_size as f32 / orig_w as f32)
        .min(target_size as f32 / orig_h as f32);

    let new_w = (orig_w as f32 * ratio).round() as u32;
    let new_h = (orig_h as f32 * ratio).round() as u32;

    // Padding demi-symétrique (comme dans la version Python : round(dX - 0.1))
    let dw = (target_size as f32 - new_w as f32) / 2.0;
    let dh = (target_size as f32 - new_h as f32) / 2.0;
    let pad_left = (dw - 0.1).round() as u32;
    let pad_top  = (dh - 0.1).round() as u32;

    // Redimensionnement de qualité Lanczos (proche d'OpenCV INTER_LINEAR)
    let resized = img
        .resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
        .to_rgb8();

    // Canvas gris
    let mut canvas = RgbImage::from_pixel(target_size, target_size, Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, pad_left as i64, pad_top as i64);

    LetterboxResult { image: canvas, ratio, pad: (pad_left, pad_top) }
}

/// Convertit une image RGB `RgbImage` en tenseur normalisé de forme [1, 3, H, W],
/// valeurs dans [0.0, 1.0] (f32).
///
/// Correspond à :
/// ```python
/// tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
/// ```
fn image_to_tensor(img: &RgbImage) -> Array4<f32> {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);

    // Collecte des pixels en mémoire contiguë HWC, puis réorganisation en NCHW
    // via ndarray (zéro copie supplémentaire grâce au réindexage).
    let mut data = vec![0.0_f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            // Canaux R, G, B séparés
            data[0 * h * w + y * w + x] = px[0] as f32 / 255.0;
            data[1 * h * w + y * w + x] = px[1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = px[2] as f32 / 255.0;
        }
    }

    // Shape [1, 3, H, W]
    Array4::from_shape_vec((1, 3, h, w), data)
        .expect("Impossible de créer le tenseur d'entrée")
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-traitement
// ─────────────────────────────────────────────────────────────────────────────

/// Convertit un rectangle format centre `(cx, cy, w, h)` en coins `(x1, y1, x2, y2)`.
#[inline]
fn cxcywh_to_xyxy(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let hw = w * 0.5;
    let hh = h * 0.5;
    (cx - hw, cy - hh, cx + hw, cy + hh)
}

/// Calcule l'IoU (Intersection over Union) entre deux boites au format xyxy.
fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);

    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    if inter == 0.0 {
        return 0.0;
    }

    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;

    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Non-Maximum Suppression sur la sortie brute de l'export ONNX.
///
/// # Format du tenseur `output`
/// Shape attendue : `[1, 4 + nc, num_anchors]`
/// - `output[0, 0..4, a]`  = (cx, cy, w, h) en pixels espace letterbox
/// - `output[0, 4.., a]`   = probabilités de classe (sigmoid, ∈ [0, 1])
///
/// # Algorithme
/// 1. Filtrage par seuil de confiance (`max_cls_score >= conf_threshold`)
/// 2. Conversion cxcywh → xyxy
/// 3. NMS indépendant par classe (Greedy NMS standard)
///
/// Retourne une liste de `Detection` dans l'espace letterbox.
fn non_max_suppression(
    output: &ArrayViewD<f32>,
    conf_threshold: f32,
    iou_threshold: f32,
    nc: usize,
) -> Result<Vec<Detection>> {
    let shape = output.shape();
    if shape.len() != 3 {
        bail!(
            "Tenseur de sortie attendu 3D [B, 4+nc, anchors], obtenu {:?}",
            shape
        );
    }
    if shape[1] < 4 + nc {
        bail!(
            "Dimension 1 du tenseur ({}) < 4 + nc ({})",
            shape[1], 4 + nc
        );
    }

    let num_anchors = shape[2];
    let mut candidates: Vec<Detection> = Vec::with_capacity(num_anchors / 8);

    // ── Étape 1 : collecte des candidats au-dessus du seuil ──────────────────
    for a in 0..num_anchors {
        // Scores de classe pour cette ancre
        let mut best_cls = 0usize;
        let mut best_score = 0.0f32;
        for c in 0..nc {
            let score = output[[0, 4 + c, a]];
            if score > best_score {
                best_score = score;
                best_cls   = c;
            }
        }

        if best_score < conf_threshold {
            continue;
        }

        let cx = output[[0, 0, a]];
        let cy = output[[0, 1, a]];
        let w  = output[[0, 2, a]];
        let h  = output[[0, 3, a]];
        let (x1, y1, x2, y2) = cxcywh_to_xyxy(cx, cy, w, h);

        candidates.push(Detection { x1, y1, x2, y2, confidence: best_score, class_id: best_cls });
    }

    // ── Étape 2 : tri global par confiance décroissante ───────────────────────
    candidates.sort_by(|a, b| {
        b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── Étape 3 : NMS par classe (Greedy NMS) ─────────────────────────────────
    // Tableau de suppression : kept[i] = false  →  la détection i est écartée
    let n = candidates.len();
    let mut suppressed = vec![false; n];

    let mut kept: Vec<Detection> = Vec::new();

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let ref_det = &candidates[i];
        kept.push(ref_det.clone());

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            // NMS par classe : ne comparer que les boites de même classe
            if candidates[j].class_id != ref_det.class_id {
                continue;
            }
            if iou(ref_det, &candidates[j]) >= iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Ok(kept)
}

/// Reprojette les boites de l'espace letterbox vers l'image d'origine.
///
/// Inverse de la transformation appliquée par `letterbox` :
///   1. Soustraction du padding
///   2. Division par le ratio
///   3. Clamping dans les bornes de l'image originale
///
/// Correspond à `scale_boxes_to_original` de `infer.py`.
fn scale_boxes_to_original(
    detections: &mut [Detection],
    ratio: f32,
    pad: (u32, u32),
    orig_size: (u32, u32), // (orig_w, orig_h)
) {
    let pad_l = pad.0 as f32;
    let pad_t = pad.1 as f32;
    let max_x = (orig_size.0 - 1) as f32;
    let max_y = (orig_size.1 - 1) as f32;

    for det in detections.iter_mut() {
        det.x1 = ((det.x1 - pad_l) / ratio).clamp(0.0, max_x);
        det.y1 = ((det.y1 - pad_t) / ratio).clamp(0.0, max_y);
        det.x2 = ((det.x2 - pad_l) / ratio).clamp(0.0, max_x);
        det.y2 = ((det.y2 - pad_t) / ratio).clamp(0.0, max_y);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendu
// ─────────────────────────────────────────────────────────────────────────────

/// Retourne la couleur RGB associée à une classe (cycle sur la palette).
#[inline]
fn class_color(class_id: usize) -> Rgb<u8> {
    Rgb(PALETTE[class_id % PALETTE.len()])
}

/// Dessine un segment horizontal entre `(x1, y)` et `(x2, y)` (inclus).
fn hline(img: &mut RgbImage, x1: u32, x2: u32, y: u32, color: Rgb<u8>) {
    let (iw, ih) = img.dimensions();
    if y >= ih { return; }
    let x2 = x2.min(iw - 1);
    for x in x1..=x2 {
        img.put_pixel(x, y, color);
    }
}

/// Dessine un segment vertical entre `(x, y1)` et `(x, y2)` (inclus).
fn vline(img: &mut RgbImage, x: u32, y1: u32, y2: u32, color: Rgb<u8>) {
    let (iw, ih) = img.dimensions();
    if x >= iw { return; }
    let y2 = y2.min(ih - 1);
    for y in y1..=y2 {
        img.put_pixel(x, y, color);
    }
}

/// Dessine un rectangle (contour uniquement) d'épaisseur `thickness` en pixels.
fn draw_rect(
    img:       &mut RgbImage,
    x1:        u32,
    y1:        u32,
    x2:        u32,
    y2:        u32,
    color:     Rgb<u8>,
    thickness: u32,
) {
    let (iw, ih) = img.dimensions();
    let x2 = x2.min(iw.saturating_sub(1));
    let y2 = y2.min(ih.saturating_sub(1));

    for t in 0..thickness {
        // Bord supérieur
        hline(img, x1, x2, y1.saturating_add(t),       color);
        // Bord inférieur
        hline(img, x1, x2, y2.saturating_sub(t),       color);
        // Bord gauche
        vline(img, x1.saturating_add(t), y1, y2,       color);
        // Bord droit
        vline(img, x2.saturating_sub(t), y1, y2,       color);
    }
}

/// Dessine les boites de détection sur une copie de l'image et la retourne.
fn annotate_image(
    orig:       &DynamicImage,
    detections: &[Detection],
    class_names: &[&str],
) -> RgbImage {
    let mut rgb = orig.to_rgb8();

    for det in detections {
        let x1 = det.x1.max(0.0) as u32;
        let y1 = det.y1.max(0.0) as u32;
        let x2 = det.x2.max(0.0) as u32;
        let y2 = det.y2.max(0.0) as u32;

        // Couleur de la classe
        let color = class_color(det.class_id);

        // Boite principale (épaisseur 2 px)
        draw_rect(&mut rgb, x1, y1, x2, y2, color, 2);

        // Petits repères de coin en L (style "futuriste" comme dans infer.py)
        let w = (x2 - x1) as f32;
        let h = (y2 - y1) as f32;
        let cl = (w.min(h) * 0.18).max(6.0) as u32; // longueur du coin

        // Top-left
        hline(&mut rgb, x1, x1 + cl, y1, color);
        vline(&mut rgb, x1, y1, y1 + cl, color);
        // Top-right
        hline(&mut rgb, x2.saturating_sub(cl), x2, y1, color);
        vline(&mut rgb, x2, y1, y1 + cl, color);
        // Bottom-left
        hline(&mut rgb, x1, x1 + cl, y2, color);
        vline(&mut rgb, x1, y2.saturating_sub(cl), y2, color);
        // Bottom-right
        hline(&mut rgb, x2.saturating_sub(cl), x2, y2, color);
        vline(&mut rgb, x2, y2.saturating_sub(cl), y2, color);

        // Log du label dans stdout (pas de rendu texte sans police vectorielle)
        let _ = (class_names, det); // déjà loggé dans main
    }

    rgb
}

// ─────────────────────────────────────────────────────────────────────────────
// Point d'entrée
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // ── 1. Chargement de l'image ──────────────────────────────────────────────
    let orig_img = image::open(&args.image)
        .with_context(|| format!("Impossible d'ouvrir l'image : {}", args.image.display()))?;
    let (orig_w, orig_h) = orig_img.dimensions();
    println!(
        "[image]  {} ({}×{})",
        args.image.display(), orig_w, orig_h
    );

    // ── 2. Pré-traitement ─────────────────────────────────────────────────────
    println!("[prep]   letterbox {}→{}×{}", orig_w.max(orig_h), args.size, args.size);
    let lb = letterbox(&orig_img, args.size);
    let tensor = image_to_tensor(&lb.image);
    // tensor shape : [1, 3, size, size]

    // ── 3. Session ONNX Runtime ───────────────────────────────────────────────
    let mut session = Session::builder()
        .context("Impossible de créer le builder ONNX Runtime")?
        .commit_from_file(&args.model)
        .with_context(|| format!("Impossible de charger le modèle : {}", args.model.display()))?;

    // Affichage des entrées/sorties du graphe (utile pour le débogage)
    // En rc.12, `Outlet` expose name() et aucun champ public de type
    for inp in session.inputs().iter() {
        println!("[model]  entrée  «{}»", inp.name());
    }
    for out in session.outputs().iter() {
        println!("[model]  sortie  «{}»", out.name());
    }

    // ── 4. Inférence ─────────────────────────────────────────────────────────
    // Solution portable au conflit ndarray : passer (forme, Vec<f32>),
    // tuple qui n'est lié à aucune version de ndarray.
    let s = args.size as usize;
    let flat_data: Vec<f32> = tensor.iter().copied().collect();
    let ort_input = OrtTensor::from_array(([1_usize, 3, s, s], flat_data))
        .context("Impossible de créer le tenseur d'entrée ORT")?;

    let outputs = session
        .run(ort::inputs!["images" => ort_input])
        .context("Erreur lors du forward ONNX")?;

    // rc.12 : try_extract_tensor::<T>() retourne Result<(&Shape, &[T])>
    // où Shape = SmallVec<[i64; N]>.  On reconstruit la vue ndarray
    // manuellement.  Le closure utilise `*d` pour désambiguïser le type
    // (évite E0282 sur le pattern `|&d|`).
    let (out_shape, out_data) = outputs["output"]
        .try_extract_tensor::<f32>()
        .context("Impossible d'extraire la sortie ONNX en f32")?;
    let dims: Vec<usize> = out_shape.iter().map(|d| *d as usize).collect();
    let output_view = ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&dims), out_data)
        .context("Impossible de construire la vue ndarray depuis la sortie ONNX")?;

    println!(
        "[infer]  sortie shape : {:?}",
        output_view.shape()
    );

    // ── 5. NMS (décodage + filtrage + suppression) ────────────────────────────
    let mut detections = non_max_suppression(
        &output_view,
        args.conf,
        args.iou,
        args.nc,
    )?;

    println!(
        "[nms]    {} candidat(s) → {} détection(s) (conf≥{:.2}, iou≤{:.2})",
        output_view.shape()[2],
        detections.len(),
        args.conf,
        args.iou,
    );

    // ── 6. Reprojection vers l'image d'origine ────────────────────────────────
    scale_boxes_to_original(&mut detections, lb.ratio, lb.pad, (orig_w, orig_h));

    // ── 7. Affichage console ──────────────────────────────────────────────────
    let class_names: &[&str] = if args.nc == 80 {
        COCO_CLASSES
    } else {
        &[] // noms génériques (class_N) si nc ≠ 80
    };

    for (i, det) in detections.iter().enumerate() {
        let name = if det.class_id < class_names.len() {
            class_names[det.class_id].to_owned()
        } else {
            format!("class_{}", det.class_id)
        };
        println!(
            "  [{i:>3}]  {name:<20}  conf={:.4}  \
             x1={:.1}  y1={:.1}  x2={:.1}  y2={:.1}",
            det.confidence, det.x1, det.y1, det.x2, det.y2
        );
    }

    // ── 8. Rendu + sauvegarde ─────────────────────────────────────────────────
    if let Some(out_path) = &args.output {
        let annotated = annotate_image(&orig_img, &detections, class_names);
        annotated
            .save(out_path)
            .with_context(|| format!("Impossible de sauvegarder : {}", out_path.display()))?;
        println!("[save]   image annotée → {}", out_path.display());
    } else {
        println!("[save]   aucun chemin de sortie fourni (--output)");
    }

    Ok(())
}
