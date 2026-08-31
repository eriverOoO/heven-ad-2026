# Planner-facing highway-merge advisory derived from one HighwayMergeGapRiskArray
# frame. This is NOT a vehicle command and NOT a maneuver trigger: it contains
# no lane-change command, steering, path, trajectory, gear, brake, throttle,
# acceleration / speed request, CtrlCmd, or DBW field. MERGE_READY means the
# available merge-prediction evidence supports an acceptable front AND rear
# target-lane gap under the explicit v1 policy -- it does NOT mean "start
# steering now". A future mission / planner layer consumes this advisory; there
# is no production consumer in v1.
#
# header.frame_id is the source HighwayMergeGapRiskArray frame (the merge
# geometry frame, map); header.stamp is the source risk stamp. Exactly one
# response is published per accepted risk frame; the advisory is never latched,
# so a consumer must apply its own freshness timeout (no greater than
# maximum_input_age_s) and treat the absence of a fresh response as "no merge
# advisory". No field is ever NaN or Inf.

uint8 ACTION_MERGE_READY=0
uint8 ACTION_WAIT=1
uint8 ACTION_HOLD=2

uint8 REASON_NONE=0
# MERGE_READY: front and rear target-lane spacing and predicted route clearance
# all satisfy the v1 policy (or there is no relevant merge object).
uint8 REASON_CLEAR_GAP=1
# Nearest object ahead at the ego merge time is inside the front headway policy.
uint8 REASON_FRONT_GAP=2
# Nearest object behind at the ego merge time is inside the rear headway policy.
uint8 REASON_REAR_GAP=3
# A relevant object currently behind the ego is closing on the ego route
# station faster than the rear closing-time policy allows.
uint8 REASON_REAR_CLOSING=4
# A relevant object is alongside the ego at the merge time (inside the
# risk-layer alongside band); it is neither a clean front nor a clean rear gap.
uint8 REASON_ALONGSIDE=5
# A relevant object's predicted minimum route-station separation from a
# constant-speed ego rollout is inside the predicted route-gap policy, even
# though the front/rear spacing at the single merge instant looks acceptable.
uint8 REASON_PREDICTED_ROUTE_CONFLICT=6
# A relevant object's prediction does not cover the ego merge time (or it
# carries no predicted route station / minimum route gap). The fallback
# constant-speed delta_s_at_merge_m is diagnostic only and never authorizes a
# merge.
uint8 REASON_INSUFFICIENT_PREDICTION=7
# The ego merge timing is invalid for this traversal (stopped / below the
# upstream ego-speed epsilon), so a merge cannot be evaluated.
uint8 REASON_INVALID_EGO_STATE=8

std_msgs/Header header

uint8 action
bool active
uint8 reason

string merge_zone_id

# The one relevant object that set a non-ready action. Zero UUID / invalid when
# the action is MERGE_READY or the frame is inactive.
bool source_object_valid
unique_identifier_msgs/UUID source_object_id

# Number of relevant_to_merge objects the policy evaluated.
uint16 relevant_object_count

# Ego context copied / derived from the source frame. ego_merge_time_s is the
# upstream constant-current-speed arrival estimate (not a planner trajectory)
# and is 0.0 when ego_merge_timing_valid is false.
float32 ego_speed_mps
float32 ego_route_distance_to_merge_m
bool ego_merge_timing_valid
float32 ego_merge_time_s

# Approach envelope for the WAIT vs HOLD boundary.
# available_distance_m = max(0, ego_route_distance_to_merge_m - merge_standoff_m).
# comfortable_stop_distance_m = ego_speed_mps^2 / (2 * comfortable_deceleration_mps2).
# WAIT while available_distance_m > comfortable_stop_distance_m, else HOLD. HOLD
# is an advisory decision-boundary state, NOT a stop command.
float32 available_distance_m
float32 comfortable_stop_distance_m

# True only when every relevant object's prediction covers the ego merge time.
# MERGE_READY requires this to be true.
bool complete_prediction_coverage

# Nearest relevant object ahead of the merge reference station at the ego merge
# time. front_gap_m is its delta_s_at_merge_m (route stations, > 0).
# front_time_headway_s = front_gap_m / max(ego_speed_mps, epsilon); it is NOT a
# TTC. All false / 0.0 when no relevant object is ahead at merge.
bool front_object_valid
unique_identifier_msgs/UUID front_object_id
float32 front_gap_m
bool front_time_headway_valid
float32 front_time_headway_s

# Nearest relevant object behind the merge reference station at the ego merge
# time. rear_gap_m is -delta_s_at_merge_m (route stations, > 0).
# rear_time_headway_s = rear_gap_m / max(rear object longitudinal speed,
# epsilon). All false / 0.0 when no relevant object is behind at merge.
bool rear_object_valid
unique_identifier_msgs/UUID rear_object_id
float32 rear_gap_m
bool rear_time_headway_valid
float32 rear_time_headway_s

# The relevant object currently behind the ego (delta_s_now_m < 0) that is
# closing on the ego route station the fastest. rear_closing_time_s is its
# time_to_route_coincidence_s (route-relative, not a raw TTC).
# rear_closing_time_valid mirrors the source validity flag. All false / 0.0 when
# no relevant object behind the ego is closing.
bool rear_closing_object_valid
unique_identifier_msgs/UUID rear_closing_object_id
bool rear_closing_time_valid
float32 rear_closing_time_s

# Smallest predicted_min_route_gap_m over all relevant objects that carry a
# valid one. This is the strongest single fact: a front / rear object separated
# at the merge instant can still come much closer on the route slightly
# before / after it.
bool predicted_route_clearance_valid
float32 minimum_predicted_route_gap_m
