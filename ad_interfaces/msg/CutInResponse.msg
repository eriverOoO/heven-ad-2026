# Planner-facing longitudinal response request derived from one CutInRiskArray
# frame. This is a REQUEST / CONSTRAINT the planner may consider, never a
# vehicle command: it contains no brake, throttle, steering, trajectory offset,
# gear, lane-change, or DBW field. requested_max_speed_mps is an upper bound the
# planner should apply to its own desired speed; it is never a commanded
# velocity and is never negative.
#
# header.frame_id is the source CutInRiskArray frame (normally the route
# corridor frame map); header.stamp is the source risk stamp. Exactly one
# response is published per accepted risk frame; the response is never latched
# beyond the configured release window, so a consumer must apply its own
# freshness timeout and treat the absence of a fresh response as "no cut-in
# longitudinal constraint".

uint8 ACTION_NONE=0
uint8 ACTION_SLOWDOWN=1
uint8 ACTION_HOLD=2

uint8 REASON_NONE=0
# SLOWDOWN: the comfortable-braking margin to the cut-in station is eroding.
uint8 REASON_APPROACHING_ENTRY=1
# HOLD: the object is within the standoff distance, or a valid TTC is below the
# maximum-braking stopping time.
uint8 REASON_COLLISION_CONFLICT=2
# Reserved for a future footprint-aware refinement; not emitted in v1.
uint8 REASON_SMALL_PREDICTED_CLEARANCE=3

std_msgs/Header header

uint8 action
bool active
uint8 reason

# Number of cut_in_candidate objects in the source frame (candidates only; a
# non-candidate object never contributes to this response).
uint16 candidate_count

# The most restrictive candidate that set the aggregate action. Zero UUID and
# SIDE_NONE when action is ACTION_NONE.
unique_identifier_msgs/UUID source_object_id
uint8 source_side

# Longitudinal request. Valid only when active. ACTION_HOLD maps to 0.0 (the
# planner should prepare to hold longitudinal progress); ACTION_SLOWDOWN maps to
# the speed from which the configured comfortable deceleration still stops the
# ego at the standoff point, floored at minimum_response_speed_mps and always
# strictly below the current ego speed.
bool requested_max_speed_valid
float32 requested_max_speed_mps

# Transparent physical decision inputs for the limiting candidate. Never NaN or
# Inf. required_deceleration_mps2 is the constant deceleration that would bring
# the ego to rest at the standoff point from its current speed; when the object
# is already within the standoff distance it is reported as maximum_deceleration
# _mps2 (a floor, not the exact value). It describes the stopping-distance tier
# only: a hold driven by TTC can carry a required_deceleration_mps2 below the
# comfortable value, since TTC, not stopping distance, set that action.
float32 ego_speed_mps
float32 required_deceleration_mps2
float32 route_s_rel_m
float32 predicted_entry_route_s_rel_m
bool predicted_entry_valid
float32 predicted_entry_time_s
float32 lateral_velocity_toward_corridor_mps
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
