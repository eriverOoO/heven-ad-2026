# MORAI 3D GT visual validation (STEP 04)

## Verdict

**HOLD for training.** The one available static scene has no confirmed gross
box transform, dimension, or yaw failure, but STEP 04 cannot be declared a
complete pass yet. There is no driving-scene bag, and only 2/33 pedestrian and
5/74 obstacle boxes contain any LiDAR point. Those classes therefore do not
provide enough visible geometry to prove their pose origin convention.

No label was changed or automatically corrected. This step did not add or run
training or CenterPoint integration.

## Scope and method

The validator analyzed all 1,764 exported frames and 2,951 boxes. It rendered
48 stratified frames (91 boxes) and the four contact sheets were manually
reviewed. The rendered set covers near/far, left/right, rotated vehicles,
multi-object frames, visible pedestrian/obstacle frames, zero/small-return
objects, and alternative-convention outliers. Coverage within the 48 frames is:

| property | rendered frames |
|---|---:|
| near / far | 16 / 25 |
| left / right | 28 / 26 |
| rotated vehicle | 17 |
| multiple objects | 20 |
| visible pedestrian / obstacle | 2 / 4 |
| zero-point / 1--5-point object | 20 / 9 |
| 0.25 m margin-sensitive | 18 |

The input has only `static_20260805_003151`; no driving scene exists under the
workspace `bags/` directory. Five empty-GT frames were also rendered as a
negative control.

Point membership is evaluated in the box's local frame:

```text
local_x =  cos(yaw) * (px-cx) + sin(yaw) * (py-cy)
local_y = -sin(yaw) * (px-cx) + cos(yaw) * (py-cy)
inside  = |local_x| <= length/2
       && |local_y| <= width/2
       && |pz-cz|   <= height/2
```

The tool also counts points under four non-mutating hypotheses: a 0.25 m box
margin, swapped length/width, negated yaw, and a vehicle box centered at the
unshifted actor/rear-axle origin. A hypothesis is flagged only when it gains at
least 5 points and at least 1.5 times the nominal count. These are review
candidates, not corrections; road points, visible-surface bias, and occlusion
can all make a wrong hypothesis score higher.

## Transform reconstruction

The exporter uses target-from-source transforms. The restored chain is:

```text
map -> odom -> base_link -> rear_axle_link -> lidar_link
```

For each cloud it obtains `map <- base_link` from the nearest ego status,
`odom <- base_link` from dynamic TF, and the two static TF edges from the bag.
The stored derived transform is:

```text
odom <- map = (odom <- base_link) * (base_link <- map)
lidar <- map = (lidar <- base_link) * (base_link <- odom) * (odom <- map)
```

Consequently the same-frame dynamic TF factors cancel algebraically and the
box projection is equivalent to `(lidar <- base_link) * (base_link <- map)`.
The point/box overlays show no common rigid translation, axis swap, or yaw-sign
error across near/far and all four quadrants.

The derived `odom <- map` itself is not smooth: translation ranges from
`[-12.208, -5.836, -39.950]` to `[20.225, 4.048, -15.277]` m and the largest
adjacent translation step is 17.372 m (0.118 s between the corresponding
clouds); the maximum adjacent yaw step is 0.0532 rad. This does not move the
exported boxes because of the exact cancellation above, but it is a provenance
warning that must be reviewed before using the stored `map_to_odom` value for
anything other than this exporter.

## Center, dimensions, and yaw

Vehicle center uses the repository's rear-axle ground origin and is shifted by

```text
forward = (wheelbase + front_overhang - rear_overhang) / 2
upward  = height / 2
```

The focused BEV renders the exported centered box in class color and the
unshifted rear-axle-origin hypothesis as a purple dashed box. For 1,245 vehicle
boxes within 40 m having at least 10 nominal points, total point counts are
289,555 nominal versus 259,068 at the unshifted origin. Their medians are 47
and 50 respectively, illustrating why a single visible face cannot determine
the center. Visual silhouette placement plus the aggregate total supports the
forward shift; it does not support reverting to the actor origin.

For the same 1,245 visible near/mid-range vehicles, nominal `length/width/yaw`
contains 289,555 points, versus 151,889 with length/width swapped and 245,255
with yaw negated. The render heading arrow uses counter-clockwise yaw from
LiDAR +X. Rotated vehicles and quadrant changes retain the expected heading and
long-axis direction. There is no systemic vehicle length/width swap or yaw-sign
problem.

Pedestrian and obstacle policies shift the source ground origin only upward by
`height/2`. The two visible pedestrian instances and five visible obstacle
instances overlap plausible returns, but that is too little evidence to prove
that the source XY origin is their geometric center. Obstacles are square in
this bag (`0.75 x 0.75 m`), so they cannot validate length/width direction.

## Sanity and point-inside-box statistics

Structural invalid boxes are **0**: 0 NaN/non-finite, 0 non-positive
dimensions, and 0 centers outside the configured ROI. No gross malformed box
was confirmed among the 91 manually rendered boxes. The automatic review list
must not be interpreted as confirmed label errors.

The union of all low-return and alternative-hypothesis checks contains 1,764
boxes in 1,107 sample frames. Within it, 1,085 boxes trigger a geometry,
alignment-margin, or convention hypothesis (rather than low-return alone).
These are the reported **unresolved/suspicious boxes**; confirmed invalid or
visually grossly misplaced boxes remain 0.

| group | min | p10 | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| all (2,951) | 0 | 0 | 13 | 152 | 563 | 4,796 |
| vehicle (2,844) | 0 | 0 | 14 | 167.4 | 595.65 | 4,796 |
| pedestrian (33) | 0 | 0 | 0 | 0 | 2.8 | 29 |
| obstacle (74) | 0 | 0 | 0 | 0 | 1.7 | 27 |
| 0--20 m | 0 | 29 | 168 | 1,401.6 | 1,632.4 | 4,796 |
| 20--40 m | 0 | 5 | 19 | 68 | 82 | 174 |
| 40--60 m | 0 | 0 | 5 | 21 | 22 | 79 |
| 60 m+ | 0 | 0 | 0 | 8 | 12.6 | 17 |

- 503 objects have zero points: vehicle 403, pedestrian 31, obstacle 69.
- 555 objects have 1--5 points: vehicle 553, obstacle 2, pedestrian 0.
- Thus 1,058/2,951 boxes are low-return (0--5 points); most are far or small,
  not automatically bad labels.
- 564 boxes are 0.25 m margin-sensitive: vehicle 549, pedestrian 2, obstacle
  13. This is a targeted alignment-review signal.
- Alternative hypotheses flag 248 length/width, 279 yaw-sign, and 510 vehicle
  actor-origin candidates. Aggregate and visual checks do not establish these
  as convention errors.
- Exported yaw spans `[-3.13763, 3.13750]` rad with no NaN or out-of-range
  values after normalization.

## Human review queue

The complete per-object queue and reasons are in
`results/perception/morai_gt_validation/human_review_samples.csv`. Priority
samples are:

| sample ID | actor/class | reason |
|---|---|---|
| `static_20260805_003151_1785857534301055180` | 3/pedestrian | strongest pedestrian evidence; 29 exact, 60 with margin |
| `static_20260805_003151_1785857534158790585` | 3/pedestrian | only other visible pedestrian; 7 exact, 60 with margin |
| `static_20260805_003151_1785857524537470611` | 2/obstacle | 0 exact but 46 with 0.25 m margin |
| `static_20260805_003151_1785857524374964243` | 2/obstacle | 0 exact but 17 with margin |
| `static_20260805_003151_1785857524233629550` | 2/obstacle | 0 exact but 33 with margin |
| `static_20260805_003151_1785857524096542286` | 2/obstacle | 0 exact but 20 with margin |
| `static_20260805_003151_1785857603483097278` | 1/vehicle | actor 1: 25 nominal, 62 swapped, 47 negated-yaw |
| `static_20260805_003151_1785857618663487405` | 44/vehicle | 10 nominal, 65 swapped, 36 negated-yaw |
| `static_20260805_003151_1785857721310570279` | 21/vehicle | 550 nominal, 1,066 at unshifted actor origin; visible-face bias check |

The first required follow-up is a driving bag with close, unobstructed
pedestrians and non-square obstacles. Reviewers should use the render PNGs and
raw points rather than applying any of the alternative hypotheses automatically.

## Outputs and commands

From the workspace root:

```bash
MPLCONFIGDIR=/tmp/heven-matplotlib python3 \
  src/heven_ad_2026/tools/dataset/visualize_morai_gt.py \
  --dataset datasets/morai_heven \
  --output src/heven_ad_2026/results/perception/morai_gt_validation \
  --sample-count 48 \
  --overwrite

python3 -m pytest -q \
  src/heven_ad_2026/tools/dataset/test_visualize_morai_gt.py \
  src/heven_ad_2026/tools/morai_dataset_exporter/test_exporter_core.py
```

Generated output:

```text
results/perception/morai_gt_validation/
├── contact_sheet_01.jpg ... contact_sheet_04.jpg
├── renders/                         # 48 full-resolution overlays
├── frame_stats.csv
├── object_stats.csv
├── human_review_samples.csv         # sample IDs plus explicit reasons
├── selected_samples.txt
└── summary.json
```
