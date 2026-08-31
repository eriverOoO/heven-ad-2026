# One roundabout conflict-timing record for a single predicted object, derived
# from its DynamicObjectRisk (identity plus discrete predicted future centroids)
# and a fixed shared conflict region. Facts only: this message has no GO /
# YIELD / STOP / HOLD / gap-accepted / safe-to-enter / risk-score field and no
# speed, brake, or steering request. A response policy applies a gap threshold
# later.
#
# All times are seconds from the source prediction stamp. No field is ever NaN
# or Inf; an undefined quantity has its *_valid flag false and value 0.0.

# Identity, copied verbatim from the source DynamicObjectRisk.
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability

# True when the object is currently inside the conflict region OR a discrete
# predicted future centroid enters it within the prediction horizon. A nearby
# object whose predicted path never intersects the region is
# relevant_to_conflict = false regardless of how close it is.
bool relevant_to_conflict

# Straight-line map distance from the current object centroid to the conflict
# region (0.0 when the centroid is inside it). Diagnostic only; it is never a
# time-to-conflict proxy.
float32 object_map_distance_to_conflict_m

# Object conflict occupancy interval [object_entry_time_s, object_exit_time_s]
# from the discrete predicted centroids. object_in_conflict_now sets
# object_entry_time_s = 0. object_exit_valid is false when the object is still
# inside the region at the end of the prediction horizon. Only the FIRST
# contiguous occupancy interval is reported: if a predicted path leaves the
# region and re-enters it later in the horizon, object_exit_time_s is the first
# departure and the re-entry is not represented in v1. A consumer must not
# assume the object is clear of the region for all time after
# object_exit_time_s.
bool object_in_conflict_now
bool object_entry_valid
float32 object_entry_time_s
bool object_exit_valid
float32 object_exit_time_s

# Signed arrival difference: object_entry_time_s - ego_entry_time_s.
#   arrival_delta_s < 0  -> object reaches the conflict region before the ego
#   arrival_delta_s > 0  -> object reaches it after the ego
#   arrival_delta_s ~ 0  -> they arrive together
# Valid only when both the ego and the object have a valid conflict entry.
bool arrival_delta_valid
float32 arrival_delta_s

# Non-negative temporal gap between the ego and object conflict occupancy
# intervals. It is 0.0 when the intervals overlap and also 0.0 when they only
# touch at a boundary. When they do not overlap it is the clear time between
# them: (ego entry time) - (object exit time) if the object clears first, or
# (object entry time) - (ego exit time) if the ego clears first. Valid only
# when both entries are valid. An object still inside the region at the horizon
# end (object_exit_valid = false) is treated as occupying it through the
# horizon.
bool temporal_gap_valid
float32 temporal_gap_s

# True when the ego and object conflict occupancy intervals strictly overlap
# (share more than a single boundary instant). occupancy_overlap == true is
# mutually consistent with temporal_gap_s == 0.0.
bool occupancy_overlap

# Generic risk facts copied unchanged from DynamicObjectRisk for consumer
# convenience. They never define relevance or the temporal gap here.
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s
