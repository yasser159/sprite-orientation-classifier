#!/usr/bin/env node
/**
 * Train a TensorFlow.js CNN to classify sprite nose orientation (N/E/S/W).
 *
 * All 108 sprites are authored nose-north. We generate 4-class data by rotating each
 * sprite 0°/90°/180°/270° via sharp. The model sees alpha-channel-only silhouettes
 * (128×128×1) — shape determines orientation, not color.
 *
 * Usage:
 *   node server/tools/train-orientation-model.js [--epochs 30] [--batch 32]
 *
 * Output:
 *   server/models/sprite-orientation/model.json + weights
 *   client/public/models/sprite-orientation/  (copy for browser loading)
 */

import { resolve, join } from 'path';
import { readdir, cp, stat } from 'fs/promises';
import sharp from 'sharp';

// Use pure @tensorflow/tfjs — tfjs-node has compatibility issues with Node 25.x
// (isNullOrUndefined removed from Node's util module). Pure JS is ~3-5x slower
// but this is a one-time training run on a tiny dataset, so it's fine.
const tf = await import('@tensorflow/tfjs');
console.log('[train] Using @tensorflow/tfjs (pure JS backend)');

// ── Paths ────────────────────────────────────────────────────────────────────
const PROJECT_ROOT = resolve(import.meta.dirname, '../..');
const SPRITES_DIR = join(PROJECT_ROOT, 'client/public/sprites');
const MODEL_DIR = join(PROJECT_ROOT, 'server/models/sprite-orientation');
const CLIENT_MODEL_DIR = join(PROJECT_ROOT, 'client/public/models/sprite-orientation');

// ── Config ───────────────────────────────────────────────────────────────────
const CLASS_LABELS = ['north', 'east', 'south', 'west'];
const ROTATIONS = [0, 90, 180, 270]; // sharp rotation degrees → class index
const IMG_SIZE = 32; // 32×32 is enough for silhouette shape — keeps pure-JS training feasible

const args = process.argv.slice(2);
const EPOCHS = Number(args.find((_, i, a) => a[i - 1] === '--epochs') || 30);
const BATCH_SIZE = Number(args.find((_, i, a) => a[i - 1] === '--batch') || 32);
const VAL_RATIO = 0.2;

console.log(`[train] Config: epochs=${EPOCHS}, batch=${BATCH_SIZE}, valRatio=${VAL_RATIO}`);

// ── Data loading ─────────────────────────────────────────────────────────────

/** Extract alpha channel from a PNG file → Float32Array[64*64], normalized [0,1] */
async function extractAlpha(filePath, rotation = 0) {
  let pipeline = sharp(filePath);
  if (rotation > 0) pipeline = pipeline.rotate(rotation);
  const raw = await pipeline.resize(IMG_SIZE, IMG_SIZE, { fit: 'fill' })
    .raw().ensureAlpha().toBuffer();

  const alpha = new Float32Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    alpha[i] = raw[i * 4 + 3] / 255.0;
  }
  return alpha;
}

/** Apply data augmentation: brightness jitter + small rotation noise */
function augmentAlpha(alpha) {
  const out = new Float32Array(alpha.length);
  // Random brightness: multiply by [0.85, 1.15]
  const brightness = 0.85 + Math.random() * 0.3;
  for (let i = 0; i < alpha.length; i++) {
    out[i] = Math.min(1.0, Math.max(0.0, alpha[i] * brightness));
  }
  return out;
}

async function loadDataset() {
  const files = (await readdir(SPRITES_DIR)).filter(f => f.endsWith('.png')).sort();
  console.log(`[train] Found ${files.length} sprites in ${SPRITES_DIR}`);

  // Sprite-level train/val split to prevent data leakage
  const shuffled = [...files].sort(() => Math.random() - 0.5);
  const splitIdx = Math.floor(files.length * (1 - VAL_RATIO));
  const trainFiles = shuffled.slice(0, splitIdx);
  const valFiles = shuffled.slice(splitIdx);
  console.log(`[train] Split: ${trainFiles.length} train sprites, ${valFiles.length} val sprites`);

  const trainX = [], trainY = [];
  const valX = [], valY = [];

  // Process training sprites (base rotations only — augmentation skipped for speed)
  for (const file of trainFiles) {
    const filePath = join(SPRITES_DIR, file);
    for (let ri = 0; ri < ROTATIONS.length; ri++) {
      const alpha = await extractAlpha(filePath, ROTATIONS[ri]);
      trainX.push(alpha);
      trainY.push(ri);
    }
  }

  // Process validation sprites (no augmentation)
  for (const file of valFiles) {
    const filePath = join(SPRITES_DIR, file);
    for (let ri = 0; ri < ROTATIONS.length; ri++) {
      const alpha = await extractAlpha(filePath, ROTATIONS[ri]);
      valX.push(alpha);
      valY.push(ri);
    }
  }

  console.log(`[train] Train samples: ${trainX.length}, Val samples: ${valX.length}`);

  // Stack into tensors
  const xTrain = tf.tensor4d(
    Float32Array.from(trainX.flatMap(a => [...a])),
    [trainX.length, IMG_SIZE, IMG_SIZE, 1],
  );
  const yTrain = tf.oneHot(tf.tensor1d(trainY, 'int32'), 4).toFloat();
  const xVal = tf.tensor4d(
    Float32Array.from(valX.flatMap(a => [...a])),
    [valX.length, IMG_SIZE, IMG_SIZE, 1],
  );
  const yVal = tf.oneHot(tf.tensor1d(valY, 'int32'), 4).toFloat();

  return { xTrain, yTrain, xVal, yVal, trainFiles, valFiles };
}

// ── Model ────────────────────────────────────────────────────────────────────
function buildModel() {
  const model = tf.sequential();

  // Lean CNN for 32×32×1 input — optimized for pure-JS training speed
  model.add(tf.layers.conv2d({
    inputShape: [IMG_SIZE, IMG_SIZE, 1],
    filters: 8, kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 })); // 16×16×8

  model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 })); // 8×8×16

  model.add(tf.layers.globalAveragePooling2d({})); // 16
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// ── Confusion matrix ─────────────────────────────────────────────────────────
async function printConfusionMatrix(model, x, y) {
  const preds = model.predict(x);
  const predClasses = (await preds.argMax(-1).data());
  const trueClasses = (await y.argMax(-1).data());
  preds.dispose();

  const matrix = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < predClasses.length; i++) {
    matrix[trueClasses[i]][predClasses[i]]++;
  }

  console.log('\n── Confusion Matrix (rows=true, cols=predicted) ──');
  console.log('           ' + CLASS_LABELS.map(l => l.padStart(7)).join(''));
  for (let r = 0; r < 4; r++) {
    const row = matrix[r].map(v => String(v).padStart(7)).join('');
    const total = matrix[r].reduce((a, b) => a + b, 0);
    const correct = matrix[r][r];
    const pct = total > 0 ? ((correct / total) * 100).toFixed(1) : '0.0';
    console.log(`  ${CLASS_LABELS[r].padEnd(8)} ${row}   ${pct}%`);
  }

  const totalCorrect = matrix.reduce((sum, row, i) => sum + row[i], 0);
  const totalSamples = predClasses.length;
  console.log(`\n  Overall accuracy: ${totalCorrect}/${totalSamples} (${((totalCorrect / totalSamples) * 100).toFixed(1)}%)\n`);
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.time('[train] Total time');

  // Load data
  console.log('[train] Loading and augmenting sprites...');
  const { xTrain, yTrain, xVal, yVal } = await loadDataset();

  // Build model
  const model = buildModel();
  model.summary();
  console.log(`[train] Total params: ${model.countParams().toLocaleString()}`);

  // Train
  console.log(`\n[train] Training for ${EPOCHS} epochs...\n`);
  await model.fit(xTrain, yTrain, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [xVal, yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `  epoch ${String(epoch + 1).padStart(3)}/${EPOCHS}` +
          `  loss=${logs.loss.toFixed(4)}  acc=${logs.acc.toFixed(4)}` +
          `  val_loss=${logs.val_loss.toFixed(4)}  val_acc=${logs.val_acc.toFixed(4)}`,
        );
      },
    },
  });

  // Confusion matrix on validation set
  console.log('\n── Validation Set Results ──');
  await printConfusionMatrix(model, xVal, yVal);

  // Confusion matrix on full dataset
  const xAll = xTrain.concat(xVal);
  const yAll = yTrain.concat(yVal);
  console.log('── Full Dataset Results ──');
  await printConfusionMatrix(model, xAll, yAll);

  // Save model — pure JS tfjs doesn't support file:// handler, so we serialize manually
  console.log(`[train] Saving model to ${MODEL_DIR}`);
  const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const { writeFile } = await import('fs/promises');

    // Write model topology + weight manifest
    const modelJSON = {
      modelTopology: artifacts.modelTopology,
      weightsManifest: [{
        paths: ['weights.bin'],
        weights: artifacts.weightSpecs,
      }],
      format: 'layers-model',
      generatedBy: 'TensorFlow.js sprite-orientation-trainer',
    };
    await writeFile(join(MODEL_DIR, 'model.json'), JSON.stringify(modelJSON));

    // Write weight data as binary
    const weightBuffer = Buffer.from(artifacts.weightData);
    await writeFile(join(MODEL_DIR, 'weights.bin'), weightBuffer);

    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));

  // Copy to client/public for browser loading
  console.log(`[train] Copying model to ${CLIENT_MODEL_DIR}`);
  await cp(MODEL_DIR, CLIENT_MODEL_DIR, { recursive: true });

  // Log file sizes
  const modelFiles = await readdir(MODEL_DIR);
  let totalBytes = 0;
  for (const f of modelFiles) {
    const s = await stat(join(MODEL_DIR, f));
    totalBytes += s.size;
    console.log(`  ${f}: ${(s.size / 1024).toFixed(1)} KB`);
  }
  console.log(`  Total: ${(totalBytes / 1024).toFixed(1)} KB`);

  // Cleanup tensors
  xTrain.dispose(); yTrain.dispose();
  xVal.dispose(); yVal.dispose();
  xAll.dispose(); yAll.dispose();

  console.timeEnd('[train] Total time');
  console.log('\n✅ Model trained and exported. Run the server to test inference.');
}

main().catch(err => {
  console.error('[train] Fatal:', err);
  process.exit(1);
});
