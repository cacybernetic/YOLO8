//! Inférence YOLOv8 ONNX en temps réel sur source vidéo (webcam ou fichier).
//!
//! Ce programme reproduit en Rust le pipeline du script `live.py`:
//!   1. Capture vidéo via `rustcv` (webcam ou fichier vidéo)
//!   2. Letterbox + normalisation en numpy-équivalent (ndarray)
//!   3. Inférence ONNX Runtime via `ort`
//!   4. Décodage + NMS par classe en Rust pur
//!   5. Reprojection des boites dans la frame originale
//!   6. Rendu "futuriste" (cadre + coins L + jauge) en manipulant les pixels
//!      directement, puis affichage via `rustcv::highgui`
//!
//! Build:
//!     cargo build --release
//!
//! Usage:
//!     cargo run --release -- --model weights/best.onnx --nc 80 --source 0
//!     cargo run --release -- --model weights/best.onnx --nc 80 --source video.mp4

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use rustcv::highgui;
use rustcv::prelude::*; // VideoCapture, Mat, etc.

// ---------------------------------------------------------------------------
// Noms de classes par défaut (COCO 80)
// ---------------------------------------------------------------------------

const CLASS_NAMES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
"truck", "boat", "traffic light", "fire hydrant", "stop sign",
"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
"donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
"toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
"vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

// ---------------------------------------------------------------------------
// Police bitmap 5×7 — ASCII imprimable (codes 32..=126, 95 glyphes)
// Encodage colonne-majeur : chaque octet = 1 colonne, bit 0 = ligne du haut.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static FONT5X7: [[u8; 5]; 95] = [
    [0x00,0x00,0x00,0x00,0x00], // ' '
    [0x00,0x00,0x5F,0x00,0x00], // '!'
    [0x00,0x07,0x00,0x07,0x00], // '"'
    [0x14,0x7F,0x14,0x7F,0x14], // '#'
    [0x24,0x2A,0x7F,0x2A,0x12], // '$'
    [0x23,0x13,0x08,0x64,0x62], // '%'
    [0x36,0x49,0x55,0x22,0x50], // '&'
    [0x00,0x05,0x03,0x00,0x00], // '\''
    [0x00,0x1C,0x22,0x41,0x00], // '('
    [0x00,0x41,0x22,0x1C,0x00], // ')'
    [0x14,0x08,0x3E,0x08,0x14], // '*'
    [0x08,0x08,0x3E,0x08,0x08], // '+'
    [0x00,0x50,0x30,0x00,0x00], // ','
    [0x08,0x08,0x08,0x08,0x08], // '-'
    [0x00,0x60,0x60,0x00,0x00], // '.'
    [0x20,0x10,0x08,0x04,0x02], // '/'
    [0x3E,0x51,0x49,0x45,0x3E], // '0'
    [0x00,0x42,0x7F,0x40,0x00], // '1'
    [0x42,0x61,0x51,0x49,0x46], // '2'
    [0x21,0x41,0x45,0x4B,0x31], // '3'
    [0x18,0x14,0x12,0x7F,0x10], // '4'
    [0x27,0x45,0x45,0x45,0x39], // '5'
    [0x3C,0x4A,0x49,0x49,0x30], // '6'
    [0x01,0x71,0x09,0x05,0x03], // '7'
    [0x36,0x49,0x49,0x49,0x36], // '8'
    [0x06,0x49,0x49,0x29,0x1E], // '9'
    [0x00,0x36,0x36,0x00,0x00], // ':'
    [0x00,0x56,0x36,0x00,0x00], // ';'
    [0x08,0x14,0x22,0x41,0x00], // '<'
    [0x14,0x14,0x14,0x14,0x14], // '='
    [0x00,0x41,0x22,0x14,0x08], // '>'
    [0x02,0x01,0x51,0x09,0x06], // '?'
    [0x32,0x49,0x79,0x41,0x3E], // '@'
    [0x7E,0x11,0x11,0x11,0x7E], // 'A'
    [0x7F,0x49,0x49,0x49,0x36], // 'B'
    [0x3E,0x41,0x41,0x41,0x22], // 'C'
    [0x7F,0x41,0x41,0x22,0x1C], // 'D'
    [0x7F,0x49,0x49,0x49,0x41], // 'E'
    [0x7F,0x09,0x09,0x09,0x01], // 'F'
    [0x3E,0x41,0x49,0x49,0x7A], // 'G'
    [0x7F,0x08,0x08,0x08,0x7F], // 'H'
    [0x00,0x41,0x7F,0x41,0x00], // 'I'
    [0x20,0x40,0x41,0x3F,0x01], // 'J'
    [0x7F,0x08,0x14,0x22,0x41], // 'K'
    [0x7F,0x40,0x40,0x40,0x40], // 'L'
    [0x7F,0x02,0x04,0x02,0x7F], // 'M'
    [0x7F,0x04,0x08,0x10,0x7F], // 'N'
    [0x3E,0x41,0x41,0x41,0x3E], // 'O'
    [0x7F,0x09,0x09,0x09,0x06], // 'P'
    [0x3E,0x41,0x51,0x21,0x5E], // 'Q'
    [0x7F,0x09,0x19,0x29,0x46], // 'R'
    [0x46,0x49,0x49,0x49,0x31], // 'S'
    [0x01,0x01,0x7F,0x01,0x01], // 'T'
    [0x3F,0x40,0x40,0x40,0x3F], // 'U'
    [0x1F,0x20,0x40,0x20,0x1F], // 'V'
    [0x3F,0x40,0x38,0x40,0x3F], // 'W'
    [0x63,0x14,0x08,0x14,0x63], // 'X'
    [0x07,0x08,0x70,0x08,0x07], // 'Y'
    [0x61,0x51,0x49,0x45,0x43], // 'Z'
    [0x00,0x7F,0x41,0x41,0x00], // '['
    [0x02,0x04,0x08,0x10,0x20], // '\'
    [0x00,0x41,0x41,0x7F,0x00], // ']'
    [0x04,0x02,0x01,0x02,0x04], // '^'
    [0x40,0x40,0x40,0x40,0x40], // '_'
    [0x00,0x01,0x02,0x04,0x00], // '`'
    [0x20,0x54,0x54,0x54,0x78], // 'a'
    [0x7F,0x48,0x44,0x44,0x38], // 'b'
    [0x38,0x44,0x44,0x44,0x20], // 'c'
    [0x38,0x44,0x44,0x48,0x7F], // 'd'
    [0x38,0x54,0x54,0x54,0x18], // 'e'
    [0x08,0x7E,0x09,0x01,0x02], // 'f'
    [0x0C,0x52,0x52,0x52,0x3E], // 'g'
    [0x7F,0x08,0x04,0x04,0x78], // 'h'
    [0x00,0x44,0x7D,0x40,0x00], // 'i'
    [0x20,0x40,0x44,0x3D,0x00], // 'j'
    [0x7F,0x10,0x28,0x44,0x00], // 'k'
    [0x00,0x41,0x7F,0x40,0x00], // 'l'
    [0x7C,0x04,0x18,0x04,0x78], // 'm'
    [0x7C,0x08,0x04,0x04,0x78], // 'n'
    [0x38,0x44,0x44,0x44,0x38], // 'o'
    [0x7C,0x14,0x14,0x14,0x08], // 'p'
    [0x08,0x14,0x14,0x18,0x7C], // 'q'
    [0x7C,0x08,0x04,0x04,0x08], // 'r'
    [0x48,0x54,0x54,0x54,0x20], // 's'
    [0x04,0x3F,0x44,0x40,0x20], // 't'
    [0x3C,0x40,0x40,0x20,0x7C], // 'u'
    [0x1C,0x20,0x40,0x20,0x1C], // 'v'
    [0x3C,0x40,0x30,0x40,0x3C], // 'w'
    [0x44,0x28,0x10,0x28,0x44], // 'x'
    [0x0C,0x50,0x50,0x50,0x3C], // 'y'
    [0x44,0x64,0x54,0x4C,0x44], // 'z'
    [0x00,0x08,0x36,0x41,0x00], // '{'
    [0x00,0x00,0x7F,0x00,0x00], // '|'
    [0x00,0x41,0x36,0x08,0x00], // '}'
    [0x10,0x08,0x08,0x10,0x08], // '~'
];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(author, version, about = "Inférence YOLOv8 ONNX en temps réel (Rust + rustcv).")]
struct Args {
    /// Chemin du modèle ONNX
    #[arg(long)]
    model: PathBuf,

    /// Nombre de classes du modèle
    #[arg(long)]
    nc: usize,

    /// Source vidéo: index webcam (entier) ou chemin fichier
    #[arg(long)]
    source: String,

    /// Taille d'entrée du modèle (carrée)
    #[arg(long, default_value_t = 640)]
    input_size: usize,

    /// Seuil de confiance
    #[arg(long, default_value_t = 0.25)]
    conf: f32,

    /// Seuil IoU pour NMS
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// Épaisseur des lignes (en pixels)
    #[arg(long, default_value_t = 2)]
    thickness: i32,

    /// Désactive l'affichage (utile pour bench headless)
    #[arg(long, default_value_t = false)]
    no_show: bool,
}

// ---------------------------------------------------------------------------
// Détection (struct simple, équivalent de la sortie postprocess en Python)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    cls_id: usize,
}

// ---------------------------------------------------------------------------
// Préprocessing (letterbox)
// ---------------------------------------------------------------------------

/// Redimensionne `frame` (BGR uint8) en un canvas carré `size x size` avec
/// préservation du ratio. Retourne le canvas, le ratio, et le padding (left, top).
///
/// Note: rustcv n'expose pas (encore) `cv::resize` directement. On fait donc
/// le resize en Rust pur via un nearest-neighbor simple. Pour de meilleures
/// performances et qualité, on pourrait utiliser le crate `image` ou
/// `fast_image_resize`. Le nearest reste acceptable pour de l'inférence
/// temps réel à condition que les ratios ne soient pas trop extrêmes.
fn letterbox(frame: &Mat, size: usize) -> (Vec<u8>, f32, (usize, usize), usize, usize) {
    let src_w = frame.cols as usize;
    let src_h = frame.rows as usize;
    let src = &frame.data; // BGR plat (h * w * 3)

    let r = (size as f32 / src_h as f32).min(size as f32 / src_w as f32);
    let new_w = ((src_w as f32) * r).round() as usize;
    let new_h = ((src_h as f32) * r).round() as usize;
    let pad_left = (size - new_w) / 2;
    let pad_top = (size - new_h) / 2;

    // Canvas rempli de gris 114 (convention YOLOv5/v8)
    let mut canvas = vec![114u8; size * size * 3];

    // Resize nearest-neighbor + placement direct dans le canvas (BGR)
    // On parcourt les pixels du canvas dans la zone "image" et on échantillonne
    // le pixel source correspondant.
    for y_dst in 0..new_h {
        // Coordonnée source (centrée pour atténuer le crénelage)
        let y_src = (((y_dst as f32) + 0.5) / r) as usize;
        let y_src = y_src.min(src_h - 1);
        for x_dst in 0..new_w {
            let x_src = (((x_dst as f32) + 0.5) / r) as usize;
            let x_src = x_src.min(src_w - 1);

            let src_off = (y_src * src_w + x_src) * 3;
            let dst_off = ((pad_top + y_dst) * size + (pad_left + x_dst)) * 3;
            canvas[dst_off]     = src[src_off];     // B
            canvas[dst_off + 1] = src[src_off + 1]; // G
            canvas[dst_off + 2] = src[src_off + 2]; // R
        }
    }

    (canvas, r, (pad_left, pad_top), src_w, src_h)
}

/// Convertit le canvas BGR uint8 en tenseur (1, 3, H, W) float32 normalisé [0, 1]
/// avec ordre des canaux RGB (le modèle a été entraîné avec PIL/RGB).
fn to_input_tensor(canvas: &[u8], size: usize) -> Array4<f32> {
    let mut tensor = Array4::<f32>::zeros((1, 3, size, size));
    // Itération sur les pixels: on permute B<->R en passant en RGB et on
    // place chaque canal dans son plan (CHW) après normalisation.
    for y in 0..size {
        for x in 0..size {
            let off = (y * size + x) * 3;
            let b = canvas[off]     as f32 / 255.0;
            let g = canvas[off + 1] as f32 / 255.0;
            let r = canvas[off + 2] as f32 / 255.0;
            tensor[[0, 0, y, x]] = r;
            tensor[[0, 1, y, x]] = g;
            tensor[[0, 2, y, x]] = b;
        }
    }
    tensor
}

// ---------------------------------------------------------------------------
// Postprocessing (décodage + NMS)
// ---------------------------------------------------------------------------

/// Décode la sortie ONNX (1, 4 + nc, num_anchors) en une liste de détections.
/// Format de sortie: x1, y1, x2, y2 dans l'espace letterbox (pixels).
fn postprocess(
    output: ndarray::ArrayViewD<f32>,
    nc: usize,
    conf_threshold: f32,
    iou_threshold: f32,
    max_det: usize,
) -> Vec<Detection> {
    let shape = output.shape();

    // --- Détection automatique de la disposition du tenseur ---
    //
    // YOLOv8 ONNX peut sortir selon l'outil d'export :
    //   • (1, 4+nc, na)  – layout "features-first"  ex: (1, 84, 8400)
    //   • (1, na, 4+nc)  – layout "anchors-first"   ex: (1, 8400, 84)
    //   • (4+nc, na)     – idem sans dimension batch
    //   • (na, 4+nc)     – idem sans dimension batch
    //
    // On choisit le layout en vérifiant quelle dimension vaut 4+nc.

    // Normalise en 3D (ajoute une dim batch fictive si absent)
    let (b0, b1, b2) = match shape.len() {
        3 => (shape[0], shape[1], shape[2]),
        2 => (1,        shape[0], shape[1]),
        n => panic!(
            "[postprocess] forme de tenseur inattendue: {:?} ({} dims).              Attendu 2 ou 3 dimensions.", shape, n
        ),
    };
    let _ = b0; // batch ignoré (on prend toujours l'index 0)

    // features_first = true  →  (batch, 4+nc, na)
    // features_first = false →  (batch, na, 4+nc)
    let features_first = if b1 == 4 + nc && b2 != 4 + nc {
        true
    } else if b2 == 4 + nc && b1 != 4 + nc {
        false
    } else if b1 == 4 + nc && b2 == 4 + nc {
        // Ambigu : les deux dims valent 4+nc (nc très petit ou modèle carré).
        // On suppose features-first par convention YOLOv8.
        eprintln!(
            "[postprocess] AVERTISSEMENT : les deux dimensions ({}, {}) valent 4+nc={}.              On suppose layout features-first.", b1, b2, 4 + nc
        );
        true
    } else {
        panic!(
            "[postprocess] Impossible de déterminer le layout : shape={:?}, nc={}, 4+nc={}.
            Vérifiez --nc ou le modèle ONNX exporté.",
            shape, nc, 4 + nc
        );
    };

    let na = if features_first { b2 } else { b1 };

    // Affichage une seule fois pour diagnostic (thread-safe)
    {
        use std::sync::OnceLock;
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            eprintln!(
                "[postprocess] shape={:?}  layout={}  na={}  nc={}",
                shape,
                if features_first { "features-first (1,4+nc,na)" } else { "anchors-first (1,na,4+nc)" },
                    na, nc
            );
        });
    }

    // Accesseurs selon le layout détecté.
    // On utilise des closures pour ne pas dupliquer la boucle.
    let feat = |anchor: usize, feat_idx: usize| -> f32 {
        if shape.len() == 3 {
            if features_first {
                output[[0, feat_idx, anchor]]
            } else {
                output[[0, anchor, feat_idx]]
            }
        } else {
            // 2D
            if features_first {
                output[[feat_idx, anchor]]
            } else {
                output[[anchor, feat_idx]]
            }
        }
    };

    let mut candidates: Vec<Detection> = Vec::with_capacity(64);
    for i in 0..na {
        // Meilleure classe
        let mut best_id = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..nc {
            let s = feat(i, 4 + c);
            if s > best_score {
                best_score = s;
                best_id = c;
            }
        }
        if best_score < conf_threshold {
            continue;
        }

        let cx = feat(i, 0);
        let cy = feat(i, 1);
        let w  = feat(i, 2);
        let h  = feat(i, 3);
        candidates.push(Detection {
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
            conf: best_score,
            cls_id: best_id,
        });
    }

    if candidates.is_empty() {
        return candidates;
    }

    nms_per_class(&mut candidates, iou_threshold);
    candidates.truncate(max_det);
    candidates
}

/// IoU entre deux boites xyxy.
#[inline]
fn iou(a: &Detection, b: &Detection) -> f32 {
    let xx1 = a.x1.max(b.x1);
    let yy1 = a.y1.max(b.y1);
    let xx2 = a.x2.min(b.x2);
    let yy2 = a.y2.min(b.y2);
    let w = (xx2 - xx1).max(0.0);
    let h = (yy2 - yy1).max(0.0);
    let inter = w * h;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter + 1e-9;
    inter / union
}

/// NMS gloutonne par classe, en place dans `dets`.
/// Algo: tri décroissant par score, puis on garde la 1re et on supprime
/// toutes celles de la même classe avec IoU > seuil.
fn nms_per_class(dets: &mut Vec<Detection>, iou_threshold: f32) {
    dets.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Detection> = Vec::with_capacity(dets.len());
    let mut suppressed = vec![false; dets.len()];

    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        let det_i = dets[i].clone();
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
            // Le NMS ne doit s'appliquer QU'entre boites de même classe.
            if dets[j].cls_id == det_i.cls_id && iou(&det_i, &dets[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
        keep.push(det_i);
    }
    *dets = keep;
}

/// Reprojette les boites depuis l'espace letterbox vers la frame originale.
fn scale_boxes(dets: &mut [Detection], ratio: f32, pad: (usize, usize),
               orig_w: usize, orig_h: usize) {
    let (pad_left, pad_top) = (pad.0 as f32, pad.1 as f32);
    let ow = (orig_w - 1) as f32;
    let oh = (orig_h - 1) as f32;
    for d in dets.iter_mut() {
        d.x1 = ((d.x1 - pad_left) / ratio).clamp(0.0, ow);
        d.y1 = ((d.y1 - pad_top) / ratio).clamp(0.0, oh);
        d.x2 = ((d.x2 - pad_left) / ratio).clamp(0.0, ow);
        d.y2 = ((d.y2 - pad_top) / ratio).clamp(0.0, oh);
    }
               }

               // ---------------------------------------------------------------------------
               // Rendu sur Mat (manipulation directe des bytes BGR)
               // ---------------------------------------------------------------------------

               /// Génère une palette HSV équirépartie en BGR (compatible OpenCV).
               fn build_color_palette(n: usize) -> Vec<(u8, u8, u8)> {
                   let mut colors = Vec::with_capacity(n);
                   for i in 0..n {
                       let h = i as f32 / n.max(1) as f32;
                       let (r, g, b) = hsv_to_rgb(h, 0.85, 0.95);
                       colors.push((b, g, r));
                   }
                   colors
               }

               /// HSV (h dans [0, 1], s/v dans [0, 1]) -> RGB (u8).
               fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
                   let i = (h * 6.0).floor() as i32;
                   let f = h * 6.0 - i as f32;
                   let p = v * (1.0 - s);
                   let q = v * (1.0 - f * s);
                   let t = v * (1.0 - (1.0 - f) * s);
                   let (r, g, b) = match i.rem_euclid(6) {
                       0 => (v, t, p),
                       1 => (q, v, p),
                       2 => (p, v, t),
                       3 => (p, q, v),
                       4 => (t, p, v),
                       _ => (v, p, q),
                   };
                   ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
               }

               /// Trace un pixel BGR à (x, y). Bornes vérifiées.
               #[inline]
               fn put_pixel(buf: &mut [u8], width: usize, height: usize,
                            x: i32, y: i32, color: (u8, u8, u8)) {
                   if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                       return;
                   }
                   let off = (y as usize * width + x as usize) * 3;
                   buf[off]     = color.0;
                   buf[off + 1] = color.1;
                   buf[off + 2] = color.2;
                            }

                            /// Trace un segment horizontal ou vertical d'épaisseur `thickness`.
                            /// On ne fait que des H/V donc Bresenham n'est pas nécessaire.
                            fn draw_hline(buf: &mut [u8], w: usize, h: usize,
                                          x1: i32, x2: i32, y: i32, thickness: i32, color: (u8, u8, u8)) {
                                let (xa, xb) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
                                let half = thickness / 2;
                                for dy in -half..=(thickness - half - 1).max(-half) {
                                    for x in xa..=xb {
                                        put_pixel(buf, w, h, x, y + dy, color);
                                    }
                                }
                                          }

                                          fn draw_vline(buf: &mut [u8], w: usize, h: usize,
                                                        y1: i32, y2: i32, x: i32, thickness: i32, color: (u8, u8, u8)) {
                                              let (ya, yb) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
                                              let half = thickness / 2;
                                              for dx in -half..=(thickness - half - 1).max(-half) {
                                                  for y in ya..=yb {
                                                      put_pixel(buf, w, h, x + dx, y, color);
                                                  }
                                              }
                                                        }

                                                        /// Rectangle filled.
                                                        fn fill_rect(buf: &mut [u8], w: usize, h: usize,
                                                                     x1: i32, y1: i32, x2: i32, y2: i32, color: (u8, u8, u8)) {
                                                            let xa = x1.max(0).min((w as i32) - 1);
                                                            let ya = y1.max(0).min((h as i32) - 1);
                                                            let xb = x2.max(0).min((w as i32) - 1);
                                                            let yb = y2.max(0).min((h as i32) - 1);
                                                            for y in ya..=yb {
                                                                for x in xa..=xb {
                                                                    let off = (y as usize * w + x as usize) * 3;
                                                                    buf[off]     = color.0;
                                                                    buf[off + 1] = color.1;
                                                                    buf[off + 2] = color.2;
                                                                }
                                                            }
                                                                     }

                                                                     /// Dessine la boite "futuriste" pour une détection.


                                                                     /// Rend une chaîne ASCII dans le buffer BGR via la police bitmap FONT5X7.
                                                                     ///
                                                                     /// * `scale` : grossissement entier (1 = 5x7 px, 2 = 10x14 px, …)
                                                                     /// * `bg`    : couleur de fond opaque derrière le texte ; `None` = transparent.
                                                                     fn draw_text(buf: &mut [u8], w: usize, h: usize,
                                                                                  x0: i32, y0: i32, text: &str, scale: i32,
                                                                                  fg: (u8,u8,u8), bg: Option<(u8,u8,u8)>) {
                                                                         const CW: i32 = 5;
                                                                         const CH: i32 = 7;
                                                                         const GAP: i32 = 1;
                                                                         let n = text.chars().count() as i32;
                                                                         if n == 0 { return; }
                                                                         if let Some(bg_color) = bg {
                                                                             let total_w = n * (CW + GAP) * scale - GAP * scale;
                                                                             let pad = 2 * scale;
                                                                             fill_rect(buf, w, h,
                                                                                       x0 - pad, y0 - pad,
                                                                                       x0 + total_w + pad, y0 + CH * scale + pad,
                                                                                       bg_color);
                                                                         }
                                                                         let mut cx = x0;
                                                                         for ch in text.chars() {
                                                                             let code = ch as usize;
                                                                             if code >= 32 && code <= 126 {
                                                                                 let glyph = &FONT5X7[code - 32];
                                                                                 for col in 0..CW {
                                                                                     let bits = glyph[col as usize];
                                                                                     for row in 0..CH {
                                                                                         if (bits >> row) & 1 != 0 {
                                                                                             for sy in 0..scale {
                                                                                                 for sx in 0..scale {
                                                                                                     put_pixel(buf, w, h,
                                                                                                               cx + col*scale + sx,
                                                                                                               y0 + row*scale + sy,
                                                                                                               fg);
                                                                                                 }
                                                                                             }
                                                                                         }
                                                                                     }
                                                                                 }
                                                                             }
                                                                             cx += (CW + GAP) * scale;
                                                                         }
                                                                                  }

                                                                                  fn draw_futuristic_box(buf: &mut [u8], w: usize, h: usize,
                                                                                                         det: &Detection, label: &str, color: (u8, u8, u8), thickness: i32) {
                                                                                      let x1 = det.x1.round() as i32;
                                                                                      let y1 = det.y1.round() as i32;
                                                                                      let x2 = det.x2.round() as i32;
                                                                                      let y2 = det.y2.round() as i32;
                                                                                      if x2 <= x1 || y2 <= y1 {
                                                                                          return;
                                                                                      }
                                                                                      let bw = x2 - x1;
                                                                                      let bh = y2 - y1;

                                                                                      // 1) Cadre fin 1px
                                                                                      draw_hline(buf, w, h, x1, x2, y1, 1, color);
                                                                                      draw_hline(buf, w, h, x1, x2, y2, 1, color);
                                                                                      draw_vline(buf, w, h, y1, y2, x1, 1, color);
                                                                                      draw_vline(buf, w, h, y1, y2, x2, 1, color);

                                                                                      // 2) Coins en L épais (longueur ~22% du min des dimensions)
                                                                                      let cl = (bw.min(bh) as f32 * 0.22).round() as i32;
                                                                                      let cl = cl.max(8);
                                                                                      // top-left
                                                                                      draw_hline(buf, w, h, x1, x1 + cl, y1, thickness, color);
                                                                                      draw_vline(buf, w, h, y1, y1 + cl, x1, thickness, color);
                                                                                      // top-right
                                                                                      draw_hline(buf, w, h, x2 - cl, x2, y1, thickness, color);
                                                                                      draw_vline(buf, w, h, y1, y1 + cl, x2, thickness, color);
                                                                                      // bottom-left
                                                                                      draw_hline(buf, w, h, x1, x1 + cl, y2, thickness, color);
                                                                                      draw_vline(buf, w, h, y2 - cl, y2, x1, thickness, color);
                                                                                      // bottom-right
                                                                                      draw_hline(buf, w, h, x2 - cl, x2, y2, thickness, color);
                                                                                      draw_vline(buf, w, h, y2 - cl, y2, x2, thickness, color);

                                                                                      // 3) Tabs aux 4 sommets
                                                                                      let tab = thickness.max(2);
                                                                                      for &(cx, cy) in &[(x1, y1), (x2, y1), (x1, y2), (x2, y2)] {
                                                                                          fill_rect(buf, w, h, cx - tab, cy - tab, cx + tab, cy + tab, color);
                                                                                      }

                                                                                      // 4) Jauge verticale à droite
                                                                                      let gauge_width: i32 = 6;
                                                                                      let offset = thickness.max(4) + 1;
                                                                                      let mut gx1 = x2 + offset;
                                                                                      let mut gx2 = gx1 + gauge_width;
                                                                                      if gx2 >= w as i32 {
                                                                                          gx2 = (x2 - offset).max(0);
                                                                                          gx1 = (gx2 - gauge_width).max(0);
                                                                                      }
                                                                                      let gy1 = y1.max(0);
                                                                                      let gy2 = y2.min((h as i32) - 1);
                                                                                      if gx2 > gx1 && gy2 > gy1 {
                                                                                          // bord de la jauge
                                                                                          draw_hline(buf, w, h, gx1, gx2, gy1, 1, color);
                                                                                          draw_hline(buf, w, h, gx1, gx2, gy2, 1, color);
                                                                                          draw_vline(buf, w, h, gy1, gy2, gx1, 1, color);
                                                                                          draw_vline(buf, w, h, gy1, gy2, gx2, 1, color);
                                                                                          // remplissage proportionnel à conf
                                                                                          let fill_h = ((gy2 - gy1) as f32 * det.conf.clamp(0.0, 1.0)).round() as i32;
                                                                                          if fill_h > 0 {
                                                                                              fill_rect(buf, w, h, gx1, gy2 - fill_h, gx2, gy2, color);
                                                                                          }
                                                                                      }

                                                                                      // === Label : nom de classe + score de confiance ===
                                                                                      let scale = (thickness / 2).max(1);
                                                                                      let label_str = format!("{} {:.0}%", label, det.conf * 100.0);
                                                                                      let text_h = (7 * scale + 4) as i32;
                                                                                      let ty = if y1 - text_h >= 0 { y1 - text_h } else { y1 + 2 };
                                                                                      let bg = (
                                                                                          (color.0 as u32 / 3) as u8,
                                                                                                (color.1 as u32 / 3) as u8,
                                                                                                (color.2 as u32 / 3) as u8,
                                                                                      );
                                                                                      draw_text(buf, w, h, x1, ty, &label_str, scale, color, Some(bg));
                                                                                                         }

                                                                                                         // ---------------------------------------------------------------------------
                                                                                                         // Source vidéo (rustcv VideoCapture)
                                                                                                         // ---------------------------------------------------------------------------

                                                                                                         /// Ouvre une source vidéo.
                                                                                                         ///
                                                                                                         /// Limitation actuelle de `rustcv` 0.1.3 : seule l'ouverture par index
                                                                                                         /// (webcam) est supportée. `VideoCapture::from_file` n'est pas (encore)
                                                                                                         /// exposé. Si tu as besoin de lire des fichiers vidéo, deux options :
                                                                                                         ///   1. Migrer vers le crate `opencv` (binding complet) qui expose
                                                                                                         ///      `VideoCapture::from_file()`.
                                                                                                         ///   2. Pré-décoder le fichier en frames PNG/JPEG via ffmpeg, puis les
                                                                                                         ///      charger une par une.
                                                                                                         fn open_capture(source: &str) -> Result<VideoCapture> {
                                                                                                             if let Ok(idx) = source.parse::<i32>() {
                                                                                                                 if idx < 0 {
                                                                                                                     return Err(anyhow!("L'index webcam doit être positif (reçu: {})", idx));
                                                                                                                 }
                                                                                                                 // VideoCapture::new attend un u32
                                                                                                                 let cap = VideoCapture::new(idx as u32)
                                                                                                                 .with_context(|| format!("Impossible d'ouvrir la webcam #{}", idx))?;
                                                                                                                 println!("[capture] Source ouverte: webcam #{}", idx);
                                                                                                                 Ok(cap)
                                                                                                             } else {
                                                                                                                 Err(anyhow!(
                                                                                                                     "L'ouverture de fichiers vidéo n'est pas supportée par rustcv 0.1.3.\n\
Seuls les indices webcam (entiers) sont acceptés.\n\
Pour lire un fichier, migre vers le crate `opencv` ou pré-décode \
le fichier en frames PNG."
                                                                                                                 ))
                                                                                                             }
                                                                                                         }

                                                                                                         // ---------------------------------------------------------------------------
                                                                                                         // Boucle principale
                                                                                                         // ---------------------------------------------------------------------------

                                                                                                         fn main() -> Result<()> {
                                                                                                             let args = Args::parse();

                                                                                                             if !args.model.exists() {
                                                                                                                 return Err(anyhow!("Modèle ONNX introuvable: {}", args.model.display()));
                                                                                                             }

                                                                                                             // --- ONNX Runtime ---
                                                                                                             // Note: les erreurs `ort` ne sont pas Send/Sync (elles contiennent des
                                                                                                             // NonNull<...>), donc on ne peut pas les propager directement avec `?`
                                                                                                             // à travers anyhow::Result. On les convertit explicitement en string.
                                                                                                             println!("[setup] Chargement du modèle ONNX...");
                                                                                                             let mut session = Session::builder()
                                                                                                             .map_err(|e| anyhow!("ort: builder() a échoué: {}", e))?
                                                                                                             .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                                                                                                             .map_err(|e| anyhow!("ort: with_optimization_level: {}", e))?
                                                                                                             .commit_from_file(&args.model)
                                                                                                             .map_err(|e| anyhow!("ort: échec d'ouverture de {}: {}", args.model.display(), e))?;

                                                                                                             // session.inputs() (méthode, pas champ) renvoie une slice d'infos
                                                                                                             // On isole le borrow dans un bloc pour qu'il soit libéré avant
                                                                                                             // la boucle où session.run() emprunte mutuellement la session.
                                                                                                             let input_name: String = {
                                                                                                                 session.inputs()[0].name().to_string()
                                                                                                             };
                                                                                                             println!("[model] {} (input: '{}', input_size={})",
                                                                                                                      args.model.display(), input_name, args.input_size);

                                                                                                             // --- Source vidéo ---
                                                                                                             let mut cap = open_capture(&args.source)?;

                                                                                                             // --- Noms de classes + couleurs ---
                                                                                                             let class_names: Vec<String> = if args.nc == CLASS_NAMES.len() {
                                                                                                                 println!("[names] CLASS_NAMES (COCO 80) utilisé");
                                                                                                                 CLASS_NAMES.iter().map(|s| s.to_string()).collect()
                                                                                                             } else {
                                                                                                                 println!("[names] noms génériques (--nc={} ne matche pas COCO)", args.nc);
                                                                                                                 (0..args.nc).map(|i| format!("class_{}", i)).collect()
                                                                                                             };
                                                                                                             let colors = build_color_palette(args.nc);

                                                                                                             // --- Boucle ---
                                                                                                             let window_name = "YOLOv8 ONNX live (Rust)";
                                                                                                             let mut frame = Mat::empty();
                                                                                                             let mut frame_count: u64 = 0;
                                                                                                             let t_start = Instant::now();

                                                                                                             println!("[run] Démarrage. Appuyez sur 'q' (113) ou ESC (27) pour quitter.");

                                                                                                             while cap.read(&mut frame)? {
                                                                                                                 if frame.is_empty() {
                                                                                                                     continue;
                                                                                                                 }
                                                                                                                 let t0 = Instant::now();

                                                                                                                 let orig_w = frame.cols as usize;
                                                                                                                 let orig_h = frame.rows as usize;

                                                                                                                 // === 1) Préprocessing ===
                                                                                                                 let (canvas, ratio, pad, _sw, _sh) = letterbox(&frame, args.input_size);
                                                                                                                 let tensor = to_input_tensor(&canvas, args.input_size);

                                                                                                                 // === 2) Inférence ONNX ===
                                                                                                                 // Conversion explicite des erreurs ort -> anyhow (cf. note plus haut).
                                                                                                                 let input_tensor_ref = TensorRef::from_array_view(&tensor)
                                                                                                                 .map_err(|e| anyhow!("ort: TensorRef::from_array_view: {}", e))?;
                                                                                                                 let outputs = session.run(ort::inputs![&input_name => input_tensor_ref])
                                                                                                                 .map_err(|e| anyhow!("ort: session.run: {}", e))?;

                                                                                                                 // try_extract_array donne directement un ArrayViewD<f32>, plus propre
                                                                                                                 // que la version raw qui renvoie un tuple (shape, data).
                                                                                                                 let output_view = outputs[0]
                                                                                                                 .try_extract_array::<f32>()
                                                                                                                 .map_err(|e| anyhow!("ort: try_extract_array: {}", e))?;

                                                                                                                 // === 3) Postprocessing (décodage + NMS) ===
                                                                                                                 let mut detections = postprocess(
                                                                                                                     output_view,
                                                                                                                     args.nc,
                                                                                                                     args.conf,
                                                                                                                     args.iou,
                                                                                                                     300,
                                                                                                                 );

                                                                                                                 // === 4) Reprojection dans la frame originale ===
                                                                                                                 scale_boxes(&mut detections, ratio, pad, orig_w, orig_h);

                                                                                                                 // === 5) Rendu sur la frame ===
                                                                                                                 // On accède directement au buffer BGR du Mat. SafeBorrow: rustcv::Mat
                                                                                                                 // expose `data: Vec<u8>` (cf. l'exemple fourni). On modifie en place.
                                                                                                                 // Cela évite une copie complète de la frame.
                                                                                                                 {
                                                                                                                     // Re-emprunter mutablement les bytes du Mat
                                                                                                                     let buf: &mut [u8] = frame.data.as_mut_slice();
                                                                                                                     for det in &detections {
                                                                                                                         let color = colors[det.cls_id % colors.len()];
                                                                                                                         draw_futuristic_box(buf, orig_w, orig_h, det, &class_names[det.cls_id], color, args.thickness);
                                                                                                                     }
                                                                                                                 }

                                                                                                                 // FPS instantané (pour log console; pas affiché en HUD car rustcv n'a
                                                                                                                 // pas de fonction texte directe; on peut l'ajouter via crate `image`
                                                                                                                 // ou `rusttype` au besoin)
                                                                                                                 let dt = t0.elapsed().as_secs_f32();
                                                                                                                 let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };

                                                                                                                 if frame_count % 30 == 0 {
                                                                                                                     println!(
                                                                                                                         "[frame {:>5}] {:>3} det | {:.1} FPS | preprocess+inference+post in {:.1} ms",
                                                                                                                         frame_count,
                                                                                                                         detections.len(),
                                                                                                                              fps,
                                                                                                                              dt * 1000.0,
                                                                                                                     );
                                                                                                                 }

                                                                                                                 // === 6) Affichage ===
                                                                                                                 if !args.no_show {
                                                                                                                     highgui::imshow(window_name, &frame)?;
                                                                                                                     // wait_key(1) renvoie le code ASCII de la touche pressée (-1 si rien).
                                                                                                                     let key = highgui::wait_key(1)?;
                                                                                                                     if key == 113 /* 'q' */ || key == 27 /* ESC */ {
                                                                                                                         println!("[exit] Quitté par l'utilisateur (touche {}).", key);
                                                                                                                         break;
                                                                                                                     }
                                                                                                                 }

                                                                                                                 frame_count += 1;
                                                                                                             }

                                                                                                             let elapsed = t_start.elapsed().as_secs_f32();
                                                                                                             let avg_fps = if elapsed > 0.0 { frame_count as f32 / elapsed } else { 0.0 };
                                                                                                             println!(
                                                                                                                 "[done] {} frames en {:.1}s (moyenne {:.1} FPS)",
                                                                                                                      frame_count, elapsed, avg_fps
                                                                                                             );

                                                                                                             Ok(())
                                                                                                         }
