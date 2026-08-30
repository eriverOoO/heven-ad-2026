# One planner-facing relative-motion / risk record for a single predicted
# object. Derived purely from a canonical ad_interfaces/PredictedObject and the
# ego state; it embeds no tracker-backend, IMM, or covariance-layout knowledge.
#
# All positions and velocities are planar and expressed in the parent
# DynamicObjectRiskArray header frame (base_link): +x forward, +y left
# (REP-103). "Relative" always means object minus ego.
#
# No field ever carries NaN or Inf. When a metric is undefined its *_valid flag
# is false and its value fields are 0.0 (not a sentinel).

# Identity, copied verbatim from the source PredictedObject.
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability

# Current relative state in base_link.
# x_rel_m / y_rel_m: object centroid offset from the ego origin.
# distance_m: planar range |r|.
# vx_rel_mps / vy_rel_mps: relative velocity along the forward / left axes.
# relative_speed_mps: |v_rel|.
float32 x_rel_m
float32 y_rel_m
float32 distance_m
float32 vx_rel_mps
float32 vy_rel_mps
float32 relative_speed_mps

# Closing metrics. Positive => the named quantity is decreasing.
# range_rate_mps: radial closing speed -(r . v_rel) / |r| (primary, sign-stable).
# longitudinal_closing_mps: forward-gap closure -sign(x_rel) * vx_rel.
float32 range_rate_mps
float32 longitudinal_closing_mps

# Time to collision. Constant-relative-velocity contact time between two
# circumscribed circles (object radius 0.5*hypot(length,width), ego radius
# hypot(half_length,half_width)). Object orientation is never used, so an
# unobserved-yaw object and a yaw-resolved object at the same physical state
# give the same TTC. ttc_s is >= 0 and 0.0 when ttc_valid is false.
bool ttc_valid
float32 ttc_s

# Closest point of approach under constant relative velocity from the current
# state. cpa_time_s is >= 0 and clamped to the CPA horizon.
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m

# Minimum centroid separation between the object's discrete predicted states
# and a constant-velocity ego rollout, over the shared prediction horizon.
# Kept separate from the kinematic CPA above.
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s

# sqrt of the larger initial_pose xy covariance eigenvalue; 0.0 when the source
# reports no position uncertainty.
float32 position_uncertainty_m
