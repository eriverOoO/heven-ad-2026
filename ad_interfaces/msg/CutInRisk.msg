# Planner-facing cut-in evidence for one DynamicObjectRisk. This is factual
# geometry only: it contains no score, brake/stop/yield request, steering
# request, or behavior transition.

uint8 SIDE_NONE=0
uint8 SIDE_LEFT=1
uint8 SIDE_RIGHT=2

# Identity and generic object confidence copied from DynamicObjectRisk.
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability

# Current centroid geometry relative to the primary active route. s is object
# route progress minus current ego route progress; d is signed left-positive.
float32 route_s_rel_m
float32 lateral_offset_m
float32 corridor_left_width_m
float32 corridor_right_width_m
bool inside_corridor
bool near_boundary
bool adjacent_region
uint8 side

# Signed Frenet lateral motion. lateral_velocity_toward_corridor_mps is
# positive toward the nearest corridor boundary and negative away from it.
float32 lateral_velocity_mps
float32 lateral_velocity_toward_corridor_mps
bool approaching_corridor
bool longitudinally_relevant

# First discrete future centroid inside the primary route corridor. Entry s is
# relative to the current ego route station. sustained means every later
# source prediction remains inside for the configured minimum duration.
bool predicted_entry_valid
float32 predicted_entry_time_s
float32 predicted_entry_route_s_rel_m
float32 predicted_entry_lateral_offset_m
bool predicted_entry_sustained

# Existing generic risk facts copied without reinterpretation.
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s

# Deterministic conjunction of explicit geometry facts; not a response policy.
bool cut_in_candidate
