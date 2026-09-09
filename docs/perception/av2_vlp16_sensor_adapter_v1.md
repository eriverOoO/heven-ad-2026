# AV2 to MORAI VLP-16-like sensor adapter v1

## 1. Motivation

This is a small, deterministic prototype for comparing a verified AV2 LiDAR
frame with the geometry expected by the MORAI VLP-16-like path. It is neither
a physical sensor simulator nor a training-data conversion pipeline. Training
remains **DO NOT TRAIN YET**.

## 2. Verified local facts and target assumptions

The HEVEN/MORAI documentation identifies the deployed stream as a CH16 /
VLP-16-compatible Velodyne path. It establishes a 16-channel target and the
HEVEN XYZIRC layout, but no local MORAI configuration with measured vertical
angles, intensity distribution, range noise, or azimuth timing was available.

The local `/home/didgang1203/datasets/av2/` storage is reserved and currently
contains AV2 motion-forecasting assets, not a verified LiDAR-frame corpus for
this prototype. It was not modified or bulk-scanned. Consequently, the
adapter intentionally accepts a caller-supplied `N x 4` LiDAR-frame array and
does not claim AV2 source-sensor geometry or generate before/after empirical
statistics yet.

The default nominal 16 elevation list is explicit in code and configurable;
it is an approximation awaiting MORAI calibration, not a claim about the
simulator's exact laser firing geometry.

## 3. Adaptation pipeline

```text
verified AV2-frame reader (future, read-only)
  -> frame normalization (caller responsibility)
  -> physical range gate
  -> elevation / nearest target-beam assignment
  -> vertical-tolerance gate
  -> per-(beam, azimuth-bin) nearest-return selection
  -> intensity policy
  -> packed XYZIRC
```

The implementation is
`tools/centerpoint_offline/av2_vlp16_adapter.py`. It does not randomly retain
25% of points. For each retained point it computes elevation with
`atan2(z, hypot(x,y))`, assigns the nearest of 16 configured target beams,
and rejects points outside a configurable tolerance. It then selects one
nearest point deterministically per `(beam, azimuth bin)`.

## 4. Canonical output

Output has HEVEN-compatible packed dtype:

| Field | Type | Offset |
| --- | --- | ---: |
| x, y, z | float32 | 0, 4, 8 |
| intensity | uint8 | 12 |
| return_type | uint8 | 13 |
| channel | uint16 | 14 |

The stride is 16 bytes. `channel` is deterministic 0–15. AV2 has no verified
equivalent return-type contract in this work, so `return_type=0` is an
explicit placeholder rather than an asserted measurement.

Intensity policies are `preserve`, `normalize`, `clip`, `zero`, and
`constant`. No policy is declared MORAI-correct until a held-out MORAI cloud
is measured.

## 5. Bounded usage

When a verified AV2 LiDAR loader is available, use at most 10–50 frames and
write only outside AV2, for example:

```bash
PYTHONPATH=tools/centerpoint_offline python3 \
  tools/centerpoint_offline/av2_vlp16_adapter.py frame.npy \
  /home/didgang1203/datasets/centerpoint/av2_vlp16_adapter_v1/frame.npy
```

The CLI assumes `frame.npy` is an already provenance-checked `(N,4)` array
with `x,y,z,intensity` in a single LiDAR frame. It does not read or mutate
the original dataset.

## 6. Statistics and limits

`cloud_statistics` reports point count, occupied channel count, 0–20/20–40/
40–60/60–80 m range counts, and mean intensity. Before/after density,
object-point, and visualization statistics are **not yet measured** because
there is no verified local AV2 LiDAR sample and no held-out MORAI cloud.

The prototype cannot model occlusion, firing order, dual returns, material
reflectivity, weather, simulator noise, or true VLP-16 ray intersections.
It must be calibrated against a held-out MORAI recording before use in any
domain-adaptation or fine-tuning decision.

## 7. Tests and training decision

Pure Python tests cover deterministic mapping, channel bounds, range crop,
azimuth selection, empty cloud handling, intensity behavior, and exact XYZIRC
packing. They do not require ROS, CUDA, TensorRT, or AV2 writes.

**Training decision: NOT YET.** First obtain a sequence-disjoint MORAI
actor-GT bag, make Autoware inference runnable, and compare real MORAI and
adapted-cloud distributions on bounded samples.
