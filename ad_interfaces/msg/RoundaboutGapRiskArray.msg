# Policy-free roundabout conflict-timing facts for one prediction frame,
# produced by ad_roundabout_gap_risk from a canonical
# ad_interfaces/DynamicObjectRiskArray, the ego route state, and a fixed
# roundabout conflict region. It reports measured / predicted temporal
# occupancy of a shared conflict region; it contains NO GO / YIELD / STOP /
# HOLD / speed / brake / steering / behaviour-transition field and NO
# accepted-gap or safe-gap decision. A response node applies a gap policy
# later.
#
# header.frame_id is the conflict-geometry frame (map); header.stamp is the
# source DynamicObjectRiskArray stamp. An empty objects list with a valid
# header is a valid "no roundabout objects" result.
std_msgs/Header header

# The conflict region these facts were computed against.
string conflict_zone_id

# Ego conflict occupancy as a constant-current-speed arrival estimate along the
# primary route (NOT a planner trajectory prediction). ego_entry_valid /
# ego_exit_valid are false when the ego speed is below the configured epsilon,
# when the ego has already passed the conflict exit on this lap (single-lap,
# forward-only), or when the route station to the conflict entry exceeds the
# configured approach bound. ego_in_conflict_now is the polygon containment of
# the current ego pose and can be true even when the timing is invalid (e.g. a
# stopped ego already inside the region).
bool ego_in_conflict_now
bool ego_entry_valid
float32 ego_entry_time_s
bool ego_exit_valid
float32 ego_exit_time_s
float32 ego_route_distance_to_entry_m
float32 ego_route_distance_to_exit_m
float32 ego_speed_mps

# Number of objects[] entries with relevant_to_conflict == true.
uint16 relevant_object_count

RoundaboutGapRisk[] objects
