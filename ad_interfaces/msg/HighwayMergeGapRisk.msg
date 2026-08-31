# One highway-merge conflict-geometry record for a single predicted object,
# derived from its DynamicObjectRisk (identity, current relative state, and
# discrete predicted future centroids) and a fixed source-grounded merge zone on
# the active route. Facts only: this message has no MERGE / WAIT / GO / YIELD /
# HOLD / lane-change / accelerate / decelerate / merge-allowed / merge-safe /
# gap-accepted / safe-gap / risk-score field and no speed, brake, throttle, or
# steering request. A response policy applies a gap threshold later.
#
# All longitudinal quantities are route stations / speeds on the target merge
# corridor (the mainline through lane), NOT base_link x and NOT a raw TTC. All
# times are seconds from the source prediction stamp. No field is ever NaN or
# Inf; an undefined quantity has its *_valid flag false and value 0.0.

# Identity, copied verbatim from the source DynamicObjectRisk.
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability

# True when the object currently occupies the target merge corridor within the
# longitudinal relevance window around the merge zone, OR a discrete predicted
# future centroid does so within the prediction horizon. An object outside the
# longitudinal window is relevant_to_merge = false regardless of how close it is
# in base_link. When relevant_to_merge is false only the relevance flags
# (object_in_target_corridor_now, predicted_to_enter_target_corridor,
# predicted_corridor_entry_*), prediction_horizon_s, and the copied generic risk
# facts are meaningful; every route-station / speed / gap field below is 0.0 /
# false (the route projection is against a station-bounded merge-corridor
# window, so a far object's numeric context would be window-edge clamped).
bool relevant_to_merge

# Current object centroid projected onto the target merge corridor centerline.
# object_route_s_m is its route station; object_lateral_offset_m is its signed
# lateral offset (+left) from that centerline. Diagnostic context for the
# longitudinal facts below.
float32 object_route_s_m
float32 object_lateral_offset_m

# object_in_target_corridor_now: the current centroid is within the corridor
# lateral half-width (plus the configured margin) at its station.
# predicted_to_enter_target_corridor: at least one predicted centroid is inside
# the corridor within the relevance window. predicted_corridor_entry_time_s is
# the first such predicted-centroid time (valid only when
# predicted_corridor_entry_valid).
bool object_in_target_corridor_now
bool predicted_to_enter_target_corridor
bool predicted_corridor_entry_valid
float32 predicted_corridor_entry_time_s

# Current longitudinal ordering on the target corridor: object route station
# minus ego route station. delta_s_now_m > 0 => object is ahead of the ego along
# the route right now.
float32 delta_s_now_m

# Object longitudinal speed = the object's current velocity resolved onto the
# target-corridor tangent at its current station (route-frame s_dot). It uses
# the object's CURRENT velocity; it is not differentiated from the discrete
# prediction. relative_longitudinal_speed_mps = object minus ego longitudinal
# speed (> 0 => the object is pulling ahead along the route).
float32 object_longitudinal_speed_mps
float32 relative_longitudinal_speed_mps

# Predicted longitudinal ordering at the ego merge time (see
# HighwayMergeGapRiskArray.ego_merge_time_s). delta_s_at_merge_m is the object's
# predicted route station at that time minus the merge reference station
# (HighwayMergeGapRiskArray.merge_reference_route_s_m). The predicted station is
# a linear interpolation of the discrete predicted centroids when they span the
# ego merge time, otherwise a constant-current-longitudinal-speed extrapolation
# from the current station (prediction_covers_merge_time then false). Valid only
# when the ego merge timing is valid.
bool delta_s_at_merge_valid
float32 delta_s_at_merge_m

# Mutually exclusive longitudinal relationship at the ego merge time, derived
# from delta_s_at_merge_m and the configured alongside band. All false when
# delta_s_at_merge_valid is false. These are spatial-ordering facts, not a
# merge decision.
bool is_ahead_at_merge
bool is_behind_at_merge
bool is_alongside_at_merge

# Longitudinal gap dynamics on the target corridor from the CURRENT relative
# longitudinal speed. longitudinal_closing_speed_mps > 0 => the ego<->object
# route-station gap is shrinking. time_to_route_coincidence_s is
# |delta_s_now_m| / longitudinal_closing_speed_mps, the time at which the object
# and ego would share a route station if the current relative longitudinal speed
# held (valid only when longitudinal_gap_closing).
bool longitudinal_gap_closing
float32 longitudinal_closing_speed_mps
bool time_to_route_coincidence_valid
float32 time_to_route_coincidence_s

# Minimum longitudinal separation on the target corridor between the object's
# discrete predicted centroids and a constant-current-speed ego route rollout,
# over the shared prediction horizon. predicted_min_route_gap_time_s is the
# predicted-centroid time at which it occurs. Valid only when the object carries
# at least one predicted centroid.
bool predicted_min_route_gap_valid
float32 predicted_min_route_gap_m
float32 predicted_min_route_gap_time_s

# Latest discrete predicted-centroid time for this object (0.0 when it carries
# none). Always finite.
float32 prediction_horizon_s

# True only when the ego merge timing is valid AND prediction_horizon_s reaches
# at least ego_merge_time_s (within a small tolerance). When false, the discrete
# prediction does not span the ego merge time, so delta_s_at_merge_m is a
# constant-speed extrapolation -- a data-coverage flag only, never a
# safe-to-merge decision.
bool prediction_covers_merge_time

# Generic risk facts copied unchanged from DynamicObjectRisk for consumer
# convenience. They never define relevance or the merge-relative geometry here.
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s
