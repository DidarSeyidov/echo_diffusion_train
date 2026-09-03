# EchoDiffusion

Audio-guided diffusion policy for trajectory generation. A robot carrying a
microphone array hears a sound source, and a conditional diffusion model
generates the `(x, y)` trajectory that drives toward it — using audio alone, or
audio fused with vision.

The core idea is that **a single direction-of-arrival measurement has no range
information**. It constrains the source to a ray, not a point. EchoDiffusion
accumulates those rays in a recursive Bayesian BEV field anchored to the
robot's pose, so rays taken from different positions intersect and the
posterior collapses from a ridge onto a blob. Driving is what makes the
estimate sharp, and the policy is conditioned on that evolving belief.

```
ODAS /ssl + /sst  ──▶  bearings (robot frame)  ──▶  Bayesian BEV field  ──┐
                                    │                                     │
                                    └────────▶  raw DoA tokens  ──────────┤──▶  fused
                        odometry / Vicon  ──▶  ego state  ────────────────┤    conditioning
                        camera (optional) ──▶  DINOv2 + ViT adapter  ─────┘         │
                                                                                    ▼
                                          noised trajectory ──▶ Conditional 1-D U-Net ──▶ (x, y) × 20
```

---

## Status

| Component | State |
|---|---|
| ROS 2 bag reading (ODAS, audio, odometry, Vicon, images) | Working, verified against `session_audio/session_01` |
| Bayesian sound BEV field + motion triangulation | Working, unit-tested |
| Audio-only diffusion policy | Working, trains end to end |
| Image branch (DINOv2 + ViT adapter) | Implemented, untested on real frames (no camera data recorded yet) |
| Real-bag supervision | **Blocked** — the current bags carry audio only, so there are no trajectory targets yet |

Because the bags have no odometry, Vicon or camera yet, a **synthetic
generator** produces episodes in the identical on-disk format so the whole
stack is trainable and testable today. Its DoA statistics are matched to the
real recording (clutter peaks, a near-vertical phantom SST track, high source
elevation) rather than invented — see [Synthetic data](#synthetic-data).

---

## Install

```bash
conda create -n echodiffusion python=3.11 -y && conda activate echodiffusion
pip install -r requirements.txt
pip install -e .
```

No ROS installation is needed. Bags are read with a built-in CDR decoder
(`echodiffusion/data/cdr.py`), so the training environment stays a plain conda
env. `timm`/`opencv` are only imported when `data.use_image: true`.

---

## Quickstart

```bash
# 1. What is actually in the bag?
python scripts/inspect_bag.py /home/zhura/datasets/echodiffusion --plot bearings.png

# 2. Synthetic episodes, so there is something to train on today
python scripts/make_synthetic_dataset.py --out data/episodes_synth

# 3. Train (audio only)
python train.py --config configs/audio_only.yaml \
    --train-dir data/episodes_synth/train --val-dir data/episodes_synth/val

# 4. Evaluate
python evaluate.py --checkpoint checkpoints/audio_only/best.pt --viz out/
```

Once the real bags carry poses:

```bash
python scripts/prepare_dataset.py --config configs/prepare.yaml
python train.py --config configs/audio_only.yaml
```

---

## The data

`inspect_bag.py` on the reference session reports:

| Topic | Type | Rate | Content |
|---|---|---|---|
| `/ssl` | `OdasSslArrayStamped` | 125 Hz | 4 potential DoAs/frame: unit `(x,y,z)` + energy |
| `/sst` | `OdasSstArrayStamped` | 125 Hz | 2 tracked sources: `id` + unit `(x,y,z)` + activity |
| `/raw` | `AudioFrame` | 31 Hz | 6 ch, 16 kHz, int16, 512 samples/frame |
| `/sss` | `AudioFrame` | 31 Hz | 4 separated source streams |

Only `/ssl` and `/sst` are consumed — ODAS has already solved localisation, and
its geometric output is far more sample-efficient than learning a front-end
from 53 s of audio. The raw waveform decoder exists and is tested, so a
GCC-PHAT/spectrogram branch can be added later without touching the reader.

**Two quirks in the reference session that the pipeline is built around:**

1. **SST track `id=2887` sits near-vertical** (elevation 78–90°) with a
   wandering azimuth — a phantom/reflection track. Azimuth is degenerate for
   near-vertical vectors, hence `bev.max_elevation_deg`.
2. **SST track `id=5` is the real source**: azimuth 85–90° (very stable),
   elevation 47–78°, active ~51% of the time. Dormant tracks retain their last
   bearing at ~zero activity, hence `data.min_sst_activity` — folding those in
   would keep re-asserting stale evidence.

### Episode format

Bags and the synthetic generator both produce the same thing:

```
<episode_dir>/
    episode.npz      # t, ssl, ssl_n, sst, sst_n, pose_odometry,
                     # pose_vicon, source_pose, image_index, image_paths
    episode.json     # readable metadata
    images/          # only when a camera was recorded
```

Two deliberate choices:

- **Both pose streams are stored**, each possibly all-NaN. `poses.source` in
  the training config selects between them, so switching odometry ↔ Vicon is a
  re-train, not a re-prepare.
- **DoA vectors are stored unrotated**, exactly as ODAS emitted them, with the
  array extrinsics applied at load time. Recalibrating the array mounting angle
  is likewise a config edit, not a dataset rebuild.

Missing streams stay NaN and the dataset skips the affected windows, which is
what makes today's audio-only bags loadable without special-casing.

---

## The Bayesian BEV field

`echodiffusion/audio/bev_field.py`. A robot-centric log-odds grid covering
±`range_m` in `x` (forward) and `y` (left).

- **Measurement update.** Each detection deposits a von Mises ridge
  `exp(κ(cos(φ−θ) − 1))` along its bearing, **uniform in range**. `κ` scales
  with the detection's energy/activity, so confident detections cut a sharper
  ridge.
- **Motion update.** `predict(delta_pose)` rigidly warps the accumulated
  log-odds into the new body frame; unknown territory enters at log-odds 0.
  Ridges from different positions therefore intersect.
- **Forgetting.** `decay_half_life_s` is the one knob that trades
  triangulation baseline against responsiveness. Too short and every ray comes
  from nearly the same place (no parallax, the posterior stays a ridge); too
  long and a source that moves leaves a stale ghost.

The network receives `(1 + history_len, H, W)`: the fused posterior plus
re-rendered instantaneous ray maps for the last few observations. Both views
matter — the posterior has range information, but its multi-second half-life
makes it slow, so the raw recent evidence is what lets the policy react to a
new detection immediately.

Measured on a simulated source at (3.0, 1.5) m over 2.5 s (`test_audio.py`):

| Robot motion | MAP error | Posterior spread |
|---|---|---|
| stationary | 3.25 m | 1.58 m |
| straight | 0.11 m | 1.17 m |
| lateral | 0.53 m | 0.96 m |

The stationary row is not a bug — it is the bearing-only ambiguity being
reported honestly. `expected_position()` returns that spread as an explicit
uncertainty readout, and `evaluate.py` reports it split by how much the robot
moved.

> **Readout note.** The centroid is computed as a softmax over **log-odds**,
> not over probabilities. Once cells pass p ≈ 0.99 the sigmoid compresses every
> remaining difference away, and a centroid taken in probability space drifts
> along the ridge instead of settling on its peak.

---

## Array calibration

The default assumes the array is **flat, z up, x forward**. The reference
session's high elevations are consistent with a speaker mounted well above the
array, not with a tilted mount — but this has not been verified against a
measured bearing.

```bash
# Speaker at a measured bearing, robot stationary:
python scripts/calibrate_array.py <session_dir> --true-bearing-deg 45

# Or, once GT poses are recorded, fit against them directly:
python scripts/calibrate_array.py <episode_dir> --from-episode
```

It fits `azimuth_offset_deg` by weighted circular mean and tests both
handedness hypotheses, printing YAML to paste into `array:`. **Do this before
trusting any trained model on real data** — a wrong offset rotates every
trajectory by a constant angle, and the training loss will look perfectly fine.

---

## Model

| Piece | Detail |
|---|---|
| BEV encoder | 3-layer strided CNN → adaptive pool to 4×4 → MLP. Shallow on purpose: the input is already a metric probability map. Pools to a grid, not a vector, so *where* the mass sits survives. |
| DoA encoder | Per-token MLP → masked mean **and** max over detections → temporal MLP. Max-pooling matters: mean alone washes out one strong detection among several clutter peaks. |
| Ego encoder | Past poses + the filter's `(x, y, spread, confidence)` readout. |
| Image encoder | Frozen DINOv2 ViT-B/14 + zero-init bottleneck adapters in the last 6 blocks + temporal attention, then attention-pooled. Only adapters train. |
| Denoiser | Conditional 1-D U-Net over the 20 waypoints, FiLM conditioning. |
| Diffusion | DDPM training (cosine schedule, 100 steps), DDIM sampling (16 steps, η=0). Optional classifier-free guidance. |

~19.2M trainable parameters audio-only (the U-Net dominates at 18.2M).

Both modalities feed the *same* fusion layer, so an audio-only and an
audio+image checkpoint differ only in that layer's width — switching modality
is a config change, not a code path.

### Auxiliary source head

A small head regresses the GT source position from the fused conditioning.
It is cheap and well-posed, and it forces the encoders to actually localise
rather than memorise "drive forward". Set `model.predict_source: false` when
no GT source pose exists.

---

## Metrics

`evaluate.py` reports ADE/FDE in metres, plus two that matter more:

- **`endpoint_bearing_err_deg`** — angle between the predicted endpoint and
  the true source direction. A policy that ignores the audio and drives
  straight ahead can post a respectable ADE while failing this badly, so this
  is the metric that actually tests whether the sound is being used.
- **`field_spread_static_m` vs `field_spread_moving_m`** — the filter's own
  uncertainty split by how much the robot moved, i.e. a direct check that
  motion is buying localisation certainty.

The diffusion MSE is measured on noise at a random timestep and barely moves
once training is underway; do not read it as progress.

> Windows where the robot has already arrived produce an endpoint at the
> origin, whose bearing is numerical noise. Both the bearing and progress
> metrics gate on a minimum endpoint displacement (`MIN_ENDPOINT_M = 0.05`);
> without it the reported bearing error was 21° against a 3.6° median, purely
> from those windows.

### Reference numbers

40 epochs on 40 synthetic train / 8 val episodes (~7 min on an RTX 5090,
10 s/epoch), audio-only, `traj_scale` 0.882 m:

| metric | value |
|---|---|
| ADE / FDE | 0.014 m / 0.032 m |
| endpoint bearing error | 4.0° mean, 2.7° median |
| bearing within 30° | 98.9% (of 1138/1440 moving windows) |
| progress ratio vs expert | 0.99 |
| field spread, static vs moving | 1.23 m → 0.98 m |

These say the pipeline is wired correctly and that the policy is using the
bearing rather than memorising an average path. They say **nothing** about
real-world performance — the supervision is simulated.

---

## Configuration

Two configs, kept diffable so an ablation is a single file swap:

- `configs/audio_only.yaml` — the default. No `timm`, no ViT allocated.
- `configs/audio_image.yaml` — adds the DINOv2 branch. Identical audio settings.
- `configs/prepare.yaml` — bag → episode conversion (topics, rates, splits).

Keys worth knowing:

| Key | Meaning |
|---|---|
| `poses.source` | `odometry` \| `vicon` \| `auto` (auto prefers Vicon). |
| `poses.fallback` | Allow the other stream when the requested one is absent. Set `false` to fail loudly instead. |
| `array.*` | Array → base rotation, azimuth offset, handedness. |
| `bev.decay_half_life_s` | Evidence half-life — the triangulation baseline. |
| `bev.max_elevation_deg` | Drops near-vertical DoAs where azimuth is degenerate. |
| `data.min_sst_activity` | Gates dormant SST tracks holding a stale bearing. |
| `data.horizon` | Waypoints. Must be divisible by `2^(len(unet.down_dims)−1)`. |
| `data.traj_scale` | Metres per normalised unit; `auto` fits the 99th percentile from the training set. |
| `diffusion.prediction_type` | `epsilon` or `sample`. |

---

## Synthetic data

`scripts/make_synthetic_dataset.py`. A proportional unicycle expert homes on a
randomly placed source, with per-episode gain/speed jitter and a lateral detour
so the trajectories are not all straight lines. The start heading is
independent of the source bearing, so many episodes begin with the source
**behind** the robot — the case audio-only guidance must solve and vision
cannot.

Emitted detections reproduce the real bag's structure: one true-bearing SSL
potential plus lower-energy clutter, an SST track that goes dormant during
silences, and the near-vertical phantom track. Vicon is the exact pose;
odometry gets accumulating drift, so flipping `poses.source` exercises a
genuinely different signal.

It is a stand-in for supervision that does not exist yet, not a benchmark.
Numbers from it say the pipeline is wired correctly — nothing about real-world
performance.

---

## Layout

```
echodiffusion/
  audio/      odas.py (bearings, extrinsics)  bev_field.py (Bayesian filter)
  data/       cdr.py  ros_messages.py  rosbag_reader.py
              episode.py  bag_to_episode.py  synthetic.py  dataset.py
  models/     encoders.py  vit_adapter.py  unet1d.py
              diffusion.py  echo_diffusion.py
  training/   trainer.py
  utils/      geometry.py  comet_logger.py
configs/      audio_only.yaml  audio_image.yaml  prepare.yaml
scripts/      inspect_bag.py  prepare_dataset.py
              make_synthetic_dataset.py  calibrate_array.py
train.py  evaluate.py  tests/
```

---

## Logging

Comet ML, with a no-op fallback so training never depends on credentials.
Configured for workspace `iana-zhura`, project `echodiffusion`.

**The API key is deliberately not in `configs/`** — those files are tracked by
git. It lives in `~/.comet.config` (mode 600), which the resolver already
checks:

```ini
[comet]
api_key = <your key>
workspace = iana-zhura
project_name = echodiffusion
```

Resolution order is `logging.comet_api_key` → `COMET_API_KEY` →
`~/.comet.config`. So training just works:

```bash
python train.py --config configs/audio_only.yaml
python train.py --config configs/audio_only.yaml --no-comet   # disable
```

### Run names

Named from the knobs that change results, so runs are distinguishable in the
Comet UI without opening them:

```
audio-only_h20_bev6.0m-hl6.0_eps-T100_vicon_s42
└ modality  │    │              │        │    └ seed
            │    │              │        └ pose source
            │    │              └ prediction type + diffusion steps
            │    └ BEV extent and evidence half-life
            └ prediction horizon
```

Tagged with `audio-only`/`audio+image`, pose source, seed, horizon, prediction
type, BEV range, half-life and elevation gate, plus `cfg-guidance` when
classifier-free guidance is on.

### Validation figures

Every validation epoch logs **`val_traj_vs_doa`**, which puts four things in
one frame so "is the policy driving where it hears the sound?" is answerable at
a glance:

| overlay | meaning |
|---|---|
| magma background | the Bayesian sound-probability posterior |
| **yellow dashed ray** | the *measured* DoA the model actually heard (opacity = detection agreement) |
| cyan curve | predicted trajectory |
| green curve | expert trajectory |
| red ✕ / arrow | GT source position (arrow when it falls outside the crop) |

The DoA ray is recovered from the DoA token tensor the model consumed — not
from ground truth — so it shows what the policy had to work with. Figures are
always mirrored to `<output_dir>/viz/` regardless of whether Comet is active.

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers CDR decoding against hand-built messages, SE(2) round-trips, ODAS
extrinsics, the field's triangulation and warp behaviour, dataset windowing and
pose-source resolution, U-Net shape invariants, and DDIM convergence against an
oracle denoiser.

---

## Next steps

1. **Record odometry + Vicon + camera** into the bags. Everything downstream is
   already wired for them.
2. **Calibrate the array** against a measured bearing before trusting real-data
   results.
3. Then train on real episodes and compare against the synthetic baseline.
4. Optional: add a GCC-PHAT / spectrogram branch over `/raw` (the decoder is
   already there and tested) for cases where ODAS's front-end drops out.
5. Add the loss between GT source localization and estimate, and measure how close predicted trajectory to the GT source
