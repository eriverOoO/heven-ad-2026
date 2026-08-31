# Planner-facing entry advisory derived from one RoundaboutGapRiskArray frame.
# This is not a vehicle command: it contains no requested speed, brake,
# throttle, steering, path, gear, DBW, or actuator field. Consumers must apply
# a freshness timeout; exactly one response is published per accepted input.

uint8 ACTION_RELEASE=0
uint8 ACTION_YIELD=1
uint8 ACTION_HOLD=2

uint8 REASON_NONE=0
uint8 REASON_CLEAR_GAP=1
uint8 REASON_OVERLAP=2
uint8 REASON_GAP_TOO_SMALL=3
uint8 REASON_INSUFFICIENT_PREDICTION=4
uint8 REASON_INVALID_EGO_STATE=5

std_msgs/Header header
uint8 action
bool active
uint8 reason

string conflict_zone_id
bool source_object_valid
unique_identifier_msgs/UUID source_object_id
uint16 relevant_object_count

float32 ego_speed_mps
float32 ego_route_distance_to_entry_m
float32 available_distance_m
float32 comfortable_stop_distance_m

bool limiting_gap_valid
float32 limiting_gap_s
bool conflict_overlap_present
bool complete_prediction_coverage
