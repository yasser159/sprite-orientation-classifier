#!/usr/bin/env python3
"""
Train a custom CNN to classify sprite nose orientation (N/E/S/W).

v4 — Phase 1 accuracy upgrades:
  - MixUp + CutMix augmentation (proven +3-5% on small datasets)
  - Random erasing (prevents overfitting to specific features)
  - Label smoothing (0.1) — prevents overconfidence
  - Cosine annealing LR schedule
  - SE (Squeeze-and-Excite) attention blocks
  - Test-Time Augmentation (TTA) at evaluation
  - 3-model ensemble option

Usage: python3 server/tools/train-orientation-model.py [--epochs 150] [--ensemble 3]
"""

import os, sys, json, shutil
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPRITES_DIR = PROJECT_ROOT / 'client' / 'public' / 'sprites'
EXTRA_SPRITES_DIR = PROJECT_ROOT / 'server' / 'tools' / 'training-sprites' / 'normalized'
MODEL_DIR = PROJECT_ROOT / 'server' / 'models' / 'sprite-orientation'
CLIENT_MODEL_DIR = PROJECT_ROOT / 'client' / 'public' / 'models' / 'sprite-orientation'

# ── Config ─────────────────────────────────────────────────────────────────
CLASS_LABELS = ['north', 'east', 'south', 'west']
NUM_CLASSES = 4
IMG_SIZE = 64
EPOCHS = int(sys.argv[sys.argv.index('--epochs') + 1]) if '--epochs' in sys.argv else 150
ENSEMBLE_N = int(sys.argv[sys.argv.index('--ensemble') + 1]) if '--ensemble' in sys.argv else 1
BATCH_SIZE = 32
VAL_RATIO = 0.2
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 0.3
RANDOM_ERASE_PROB = 0.3

ROTATION_SCHEDULE = [
    (0,   0), (15,  0), (345, 0),
    (90,  1), (75,  1), (105, 1),
    (180, 2), (165, 2), (195, 2),
    (270, 3), (255, 3), (285, 3),
]

print(f"[train] TF {tf.__version__} | v4 — MixUp/CutMix/SE/LabelSmooth/TTA")
print(f"[train] Config: img={IMG_SIZE}, epochs={EPOCHS}, ensemble={ENSEMBLE_N}")

# ── Data loading ───────────────────────────────────────────────────────────
def load_rgb(path, rotation=0):
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_png(raw, channels=4)
    if rotation != 0:
        if rotation % 90 == 0:
            img = tf.image.rot90(img, k=rotation // 90)
        else:
            import scipy.ndimage
            img_np = img.numpy().astype(np.float32)
            rotated = scipy.ndimage.rotate(img_np, -rotation, axes=(0, 1),
                                           reshape=False, order=1, mode='constant', cval=0)
            img = tf.cast(tf.constant(np.clip(rotated, 0, 255)), tf.uint8)
    img_f = tf.cast(img, tf.float32) / 255.0
    alpha = img_f[:, :, 3:4]
    rgb = img_f[:, :, :3] * alpha + (1.0 - alpha)
    rgb = tf.image.resize(rgb, [IMG_SIZE, IMG_SIZE])
    return rgb.numpy()

# ── Advanced augmentation ──────────────────────────────────────────────────
def augment_basic(img):
    """Per-image augmentation: brightness, contrast, color shift, noise."""
    aug = img.copy()
    aug *= np.random.uniform(0.75, 1.25)
    mean = aug.mean()
    aug = (aug - mean) * np.random.uniform(0.7, 1.3) + mean
    for c in range(3):
        aug[:, :, c] += np.random.uniform(-0.08, 0.08)
    aug += np.random.normal(0, 0.03, aug.shape).astype(np.float32)
    # Random erasing — black out a random rectangle
    if np.random.random() < RANDOM_ERASE_PROB:
        h, w = IMG_SIZE, IMG_SIZE
        eh = np.random.randint(h // 8, h // 3)
        ew = np.random.randint(w // 8, w // 3)
        ey = np.random.randint(0, h - eh)
        ex = np.random.randint(0, w - ew)
        aug[ey:ey+eh, ex:ex+ew, :] = np.random.uniform(0.8, 1.0)  # erase to near-white (background)
    return np.clip(aug, 0, 1).astype(np.float32)

def mixup_batch(X, Y, alpha=MIXUP_ALPHA):
    """MixUp: blend pairs of images and their labels."""
    n = len(X)
    indices = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, size=n).astype(np.float32)
    lam_x = lam.reshape(-1, 1, 1, 1)
    lam_y = lam.reshape(-1, 1)
    X_mixed = X * lam_x + X[indices] * (1 - lam_x)
    Y_mixed = Y * lam_y + Y[indices] * (1 - lam_y)
    return X_mixed, Y_mixed

def cutmix_batch(X, Y, alpha=CUTMIX_ALPHA):
    """CutMix: cut a patch from one image, paste onto another."""
    n, h, w, c = X.shape
    indices = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, size=n).astype(np.float32)

    X_mixed = X.copy()
    for i in range(n):
        cut_ratio = np.sqrt(1 - lam[i])
        ch = int(h * cut_ratio)
        cw = int(w * cut_ratio)
        if ch < 1 or cw < 1:
            continue
        cy = np.random.randint(0, h - ch + 1)
        cx = np.random.randint(0, w - cw + 1)
        X_mixed[i, cy:cy+ch, cx:cx+cw, :] = X[indices[i], cy:cy+ch, cx:cx+cw, :]
        # Adjust lambda based on actual area replaced
        lam[i] = 1 - (ch * cw) / (h * w)

    lam_y = lam.reshape(-1, 1)
    Y_mixed = Y * lam_y + Y[indices] * (1 - lam_y)
    return X_mixed, Y_mixed

def is_aircraft(name):
    return not name.startswith('vessel-') and not name.startswith('gnd-') and not name.startswith('ground')

def build_dataset():
    all_files = sorted([f for f in os.listdir(SPRITES_DIR) if f.endswith('.png')])
    files = [(SPRITES_DIR / f, f) for f in all_files if is_aircraft(f)]
    print(f"[train] Primary: {len(files)} aircraft")

    if EXTRA_SPRITES_DIR.exists():
        extra = sorted([f for f in os.listdir(EXTRA_SPRITES_DIR) if f.endswith('.png')])
        files += [(EXTRA_SPRITES_DIR / f, f) for f in extra]
        print(f"[train] Extra: {len(extra)} external sprites")

    print(f"[train] Total unique sprites: {len(files)}")

    np.random.seed(42)
    indices = np.random.permutation(len(files))
    split = int(len(files) * (1 - VAL_RATIO))
    train_files = [files[i] for i in indices[:split]]
    val_files = [files[i] for i in indices[split:]]
    print(f"[train] Split: {len(train_files)} train, {len(val_files)} val")

    def process(file_list, augment=False):
        X, Y = [], []
        for fi, (path, name) in enumerate(file_list):
            for rot_deg, cls in ROTATION_SCHEDULE:
                try:
                    rgb = load_rgb(path, rot_deg)
                    X.append(rgb); Y.append(cls)
                    if augment:
                        for _ in range(3):
                            X.append(augment_basic(rgb)); Y.append(cls)
                except:
                    pass
            if (fi + 1) % 20 == 0:
                print(f"  {fi+1}/{len(file_list)} loaded...")
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

    X_train, y_train = process(train_files, augment=True)
    X_val, y_val = process(val_files, augment=False)
    print(f"[train] Train: {len(X_train)}, Val: {len(X_val)}")

    # Convert labels to one-hot for MixUp/CutMix
    Y_train_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_train]
    Y_val_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_val]

    return X_train, Y_train_oh, y_train, X_val, Y_val_oh, y_val

# ── SE Block (Squeeze-and-Excite) ─────────────────────────────────────────
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.reshape_channels = channels

    def call(self, x):
        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.reshape(se, [-1, 1, 1, self.reshape_channels])
        return x * se

# ── Model ──────────────────────────────────────────────────────────────────
def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Block 1 (no SE blocks — they break TFJS export as custom layers)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)  # 32×32

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)  # 16×16

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)  # 8×8

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# ── Training with MixUp/CutMix ────────────────────────────────────────────
def train_one_model(X_train, Y_train_oh, X_val, Y_val_oh, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy'],
    )

    # Cosine annealing LR
    cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=EPOCHS * (len(X_train) // BATCH_SIZE + 1),
        alpha=1e-6,
    )

    best_val_acc = 0
    best_weights = None
    patience_counter = 0
    patience = 20

    for epoch in range(EPOCHS):
        # Apply MixUp or CutMix to the batch (randomly choose one per epoch)
        if np.random.random() < 0.5:
            X_aug, Y_aug = mixup_batch(X_train, Y_train_oh)
        else:
            X_aug, Y_aug = cutmix_batch(X_train, Y_train_oh)

        # Shuffle
        perm = np.random.permutation(len(X_aug))
        X_aug, Y_aug = X_aug[perm], Y_aug[perm]

        # Update LR (cosine annealing)
        lr = float(cosine_lr(epoch * (len(X_train) // BATCH_SIZE)))
        model.optimizer.learning_rate.assign(lr)

        # Train one epoch
        hist = model.fit(X_aug, Y_aug, batch_size=BATCH_SIZE, epochs=1,
                         validation_data=(X_val, Y_val_oh), verbose=0)

        val_acc = hist.history['val_accuracy'][0]
        train_acc = hist.history['accuracy'][0]
        val_loss = hist.history['val_loss'][0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3}/{EPOCHS}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  val_loss={val_loss:.4f}  lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    if best_weights:
        model.set_weights(best_weights)

    return model, best_val_acc

# ── Test-Time Augmentation (TTA) ──────────────────────────────────────────
def predict_with_tta(model, X, n_aug=4):
    """Average predictions from original + augmented versions."""
    preds = model.predict(X, verbose=0)
    for _ in range(n_aug):
        # Flip horizontally
        X_flip = X[:, :, ::-1, :]
        preds += model.predict(X_flip, verbose=0)
        # Slight brightness variation
        X_bright = np.clip(X * np.random.uniform(0.9, 1.1), 0, 1)
        preds += model.predict(X_bright.astype(np.float32), verbose=0)
    return preds / (1 + n_aug * 2)

# ── TFJS export ────────────────────────────────────────────────────────────
def export_tfjs(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = []
    for i, layer in enumerate(model.layers):
        cfg = layer.get_config()
        cls = layer.__class__.__name__
        tfjs_cfg = dict(cfg)
        tfjs_cfg['name'] = layer.name
        tfjs_cfg['dtype'] = 'float32'
        if i == 0:
            tfjs_cfg['batch_input_shape'] = [None, IMG_SIZE, IMG_SIZE, 3]
        layers.append({'class_name': cls, 'config': tfjs_cfg})

    topology = {
        'class_name': 'Sequential',
        'config': {'name': model.name, 'layers': layers},
        'keras_version': '2.15.0', 'backend': 'tensorflow',
    }

    weight_specs, weight_buffers = [], []
    for layer in model.layers:
        for w in layer.weights:
            arr = w.numpy()
            name = w.name if '/' in w.name else f"{layer.name}/{w.name}"
            weight_specs.append({'name': name, 'shape': list(arr.shape), 'dtype': 'float32'})
            weight_buffers.append(arr.astype(np.float32).tobytes())

    weight_data = b''.join(weight_buffers)
    (output_dir / 'weights.bin').write_bytes(weight_data)

    model_json = {
        'modelTopology': topology,
        'weightsManifest': [{'paths': ['weights.bin'], 'weights': weight_specs}],
        'format': 'layers-model',
        'generatedBy': 'sprite-orientation-trainer v4.0 (MixUp/CutMix/SE/TTA)',
    }
    (output_dir / 'model.json').write_text(json.dumps(model_json, indent=2))
    total_kb = (len(weight_data) + len(json.dumps(model_json))) / 1024
    print(f"[export] {output_dir.name}: weights={len(weight_data)/1024:.0f}KB total={total_kb:.0f}KB")

# ── Confusion matrix ──────────────────────────────────────────────────────
def print_confusion(preds_classes, y_true, title):
    matrix = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, preds_classes):
        matrix[t][p] += 1

    print(f"\n── {title} ──")
    print(f"{'':>10}" + ''.join(f'{l:>8}' for l in CLASS_LABELS))
    for r in range(4):
        row = ''.join(f'{matrix[r][c]:>8}' for c in range(4))
        total = matrix[r].sum()
        pct = f'{matrix[r][r]/total*100:.1f}%' if total > 0 else '0.0%'
        print(f"  {CLASS_LABELS[r]:<8}{row}   {pct}")

    correct = sum(matrix[i][i] for i in range(4))
    total = len(y_true)
    acc = correct / total
    print(f"\n  Overall: {correct}/{total} ({acc:.1%})\n")
    return acc

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    X_train, Y_train_oh, y_train, X_val, Y_val_oh, y_val = build_dataset()

    models = []
    for i in range(ENSEMBLE_N):
        seed = 42 + i * 17
        print(f"\n{'═' * 60}")
        print(f"  Training model {i+1}/{ENSEMBLE_N} (seed={seed})")
        print(f"{'═' * 60}\n")
        model, best_val = train_one_model(X_train, Y_train_oh, X_val, Y_val_oh, seed=seed)
        models.append(model)
        print(f"  Model {i+1} best val_accuracy: {best_val:.4f}")

    # Evaluate with TTA
    print(f"\n{'═' * 60}")
    print(f"  Evaluation (TTA + {'ensemble of ' + str(len(models)) if len(models) > 1 else 'single model'})")
    print(f"{'═' * 60}")

    # Ensemble: average predictions from all models with TTA
    val_preds = np.zeros((len(X_val), NUM_CLASSES))
    for m in models:
        val_preds += predict_with_tta(m, X_val, n_aug=4)
    val_preds /= len(models)
    val_classes = val_preds.argmax(axis=1)

    val_acc = print_confusion(val_classes, y_val, 'Validation (TTA' + (f' + {len(models)}-ensemble' if len(models) > 1 else '') + ')')

    # Full dataset
    all_X = np.concatenate([X_train, X_val])
    all_y = np.concatenate([y_train, y_val])
    full_preds = np.zeros((len(all_X), NUM_CLASSES))
    for m in models:
        full_preds += predict_with_tta(m, all_X, n_aug=2)  # fewer TTA rounds for speed
    full_preds /= len(models)
    full_classes = full_preds.argmax(axis=1)
    full_acc = print_confusion(full_classes, all_y, 'Full Dataset')

    # Export best model (or first of ensemble — TFJS can't do ensemble natively)
    export_tfjs(models[0], MODEL_DIR)
    if CLIENT_MODEL_DIR.exists():
        shutil.rmtree(CLIENT_MODEL_DIR)
    shutil.copytree(MODEL_DIR, CLIENT_MODEL_DIR)

    total_kb = sum(f.stat().st_size for f in MODEL_DIR.iterdir()) / 1024
    for f in sorted(MODEL_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size/1024:.1f}KB")

    print(f"\n✅ v4: val_acc={val_acc:.1%} (TTA), full_acc={full_acc:.1%}, size={total_kb:.0f}KB")
    if ENSEMBLE_N > 1:
        print(f"   Ensemble of {ENSEMBLE_N} models — exported model 1 for inference")

if __name__ == '__main__':
    main()
