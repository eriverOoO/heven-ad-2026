# Policy-free highway-merge conflict-geometry facts for one prediction frame,
# produced by ad_highway_merge_gap_risk from a canonical
# ad_interfaces/DynamicObjectRiskArray, the ego route state, and a fixed
# source-grounded merge zone on the active route. It reports measured / predicted
# route-relative longitudinal ordering and gaps on the target merge corridor; it
# contains NO MERGE / WAIT / GO / YIELD / HOLD / RELEASE / lane-change /
# accelerate / decelerate / merge-allowed / merge-safe / gap-accepted / safe-gap
# / requested-speed / brake / throttle / steering / risk-score field. A response
# node applies a gap policy later.
#
# header.frame_id is the merge-geometry frame (map); header.stamp is the source
# DynamicObjectRiskArray stamp. An empty objects list with a valid header is a
# valid "no merge-corridor objects" result.
std_msgs/Header header

# The merge zone these facts were computed against, and the two lanes that
# define it. target_lane_sequence_id is the mainline through lane the ego merges
# into (its route stations are the longitudinal reference); source_lane_sequence_id
# is the tapering acceleration lane that bounds the merge zone.
string merge_zone_id
string target_lane_sequence_id
string source_lane_sequence_id

# Merge zone route stations on the target corridor. merge_zone_entry_route_s_m is
# where the acceleration lane begins; merge_reference_route_s_m is where it ends
# (the station by which the merge is physically resolved), against which
# HighwayMergeGapRisk.delta_s_at_merge_m is measured.
float32 merge_zone_entry_route_s_m
float32 merge_reference_route_s_m

# Ego route state on the target corridor.
float32 ego_route_s_m
float32 ego_longitudinal_speed_mps
float32 ego_route_distance_to_zone_entry_m
float32 ego_route_distance_to_merge_m

# ego_in_merge_zone_now: the ego route station is between the zone entry and the
# merge reference station. ego_merge_timing_valid is false when the ego
# longitudinal speed is below the configured epsilon, when the ego has already
# passed the merge reference station on this lap (single-lap, forward-only), or
# when the route station to the zone entry exceeds the configured approach
# bound. ego_merge_time_s is the constant-current-speed arrival estimate onto
# the merge reference station (NOT a planner trajectory) and is 0.0 when the
# timing is invalid.
bool ego_in_merge_zone_now
bool ego_merge_timing_valid
float32 ego_merge_time_s

# Number of objects[] entries with relevant_to_merge == true.
uint16 relevant_object_count

# Nearest relevant object ahead of / behind the merge reference station at the
# ego merge time (smallest |delta_s_at_merge_m| on each side; ties broken by the
# lexicographically smallest object UUID). merge_gap_m is
# nearest_leading_delta_s_at_merge_m - nearest_trailing_delta_s_at_merge_m, the
# longitudinal span on the target corridor between those two objects at the ego
# merge time; merge_gap_valid only when both a leading and a trailing object
# exist. These are measured spans, not an accepted or safe gap. A consumer
# wanting the ego's own longitudinal clearance at the merge reads
# nearest_leading_delta_s_at_merge_m / nearest_trailing_delta_s_at_merge_m
# individually (the ego sits at delta 0 by construction), not merge_gap_m.
bool nearest_leading_valid
unique_identifier_msgs/UUID nearest_leading_object_id
float32 nearest_leading_delta_s_at_merge_m
bool nearest_trailing_valid
unique_identifier_msgs/UUID nearest_trailing_object_id
float32 nearest_trailing_delta_s_at_merge_m
bool merge_gap_valid
float32 merge_gap_m

HighwayMergeGapRisk[] objects
