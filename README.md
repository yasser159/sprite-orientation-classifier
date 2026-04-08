# Sprite Orientation Classifier

A TensorFlow.js CNN that classifies top-down sprite nose orientation into 4 cardinal directions (N/E/S/W). Built for the flight-tracker project's admin QA pipeline.

## Model Stats (v4 — Phase 2)

| Metric | Value |
|--------|-------|
| Architecture | Custom CNN (Conv2D×5 + BN + GAP + Dense) |
| Input | 64×64×3 RGB |
| Output | 4-class softmax (north/east/south/west) |
| Parameters | 148,836 |
| Model size | 945KB (model.json + weights.bin) |
| Training sprites | 228 unique (73 aircraft + 155 external) |
| Val accuracy | 82.6% |
| Our aircraft accuracy | 94.5% |
| OGA blind test | 100% |
| Synthetic style resilience | 73.3% |

## Directory Structure

```
sprite-orientation-classifier/
├── model/                    # TFJS LayersModel (loadable in Node + browser)
│   ├── model.json
│   └── weights.bin
├── training-data/
│   ├── primary/              # 108 sprites from flight-tracker (Blender 3D renders)
│   ├── external/             # 155 sprites (OpenGameArt planes + Kenney vehicles)
│   └── synthetic/            # 120 style-transform test sprites (edges, posterize, etc.)
├── scripts/
│   ├── train-orientation-model.py   # Python training script (TF 2.19+)
│   └── train-orientation-model.js   # Node.js training script (slower, pure JS)
├── sprite-registry.json             # Per-sprite orientation registry (human-approved)
├── spriteRegistry.js                # Client-side JS export of registry
└── README.md
```

## Training

```bash
# Python (recommended — 30s on CPU)
python3 scripts/train-orientation-model.py --epochs 150

# Node.js (slow — hours on CPU, use only if Python unavailable)
node scripts/train-orientation-model.js --epochs 20
```

### Training pipeline
1. Load all PNGs from `training-data/primary/` + `training-data/external/`
2. Filter aircraft-only (skip vessel/ground sprites)
3. Generate 4-class data via rotation: 0°→N, 90°→E, 180°→S, 270°→W
4. Fine-grained rotations: ±15° per cardinal direction (12 rotations/sprite)
5. Augmentation: MixUp, CutMix, brightness/contrast/noise, random erasing
6. Label smoothing (0.1), cosine annealing LR
7. Export to TFJS LayersModel format

## Inference

### Node.js
```javascript
const tf = require('@tensorflow/tfjs');
const modelJSON = JSON.parse(fs.readFileSync('model/model.json', 'utf8'));
const weightData = fs.readFileSync('model/weights.bin');
const model = await tf.loadLayersModel(tf.io.fromMemory({
  modelTopology: modelJSON.modelTopology,
  weightSpecs: modelJSON.weightsManifest[0].weights,
  weightData: weightData.buffer,
}));
// Input: 64×64×3 RGB tensor, normalized [0,1], white background composite
const pred = model.predict(tf.tensor4d(rgbData, [1, 64, 64, 3]));
const labels = ['north', 'east', 'south', 'west'];
const direction = labels[pred.argMax(-1).dataSync()[0]];
```

### Browser
```javascript
const model = await tf.loadLayersModel('/models/sprite-orientation/model.json');
```

## License
- Model + scripts: MIT
- Primary sprites: project-specific (not redistributable)
- External sprites: CC0 (Kenney.nl, OpenGameArt.org)
