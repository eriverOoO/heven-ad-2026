"""Keep simulator-independent HEVEN interface contracts stable."""

from pathlib import Path

import pytest

from ad_interfaces.msg import (
    CutInResponse,
    CutInRisk,
    CutInRiskArray,
    DynamicObjectRisk,
    DynamicObjectRiskArray,
    DynamicObjectRiskState,
    RoundaboutGapRisk,
    RoundaboutGapRiskArray,
    RoundaboutGapResponse,
    HighwayMergeGapRisk,
    HighwayMergeGapRiskArray,
    HighwayMergeGapResponse,
    DynamicObstacleStatus,
    PlannerStatus,
    PredictedObject,
    PredictedObjectArray,
    PredictedState,
    TrafficLightStatus,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PREDICTION_DECLARATIONS = {
    "PredictedState.msg": """\
builtin_interfaces/Duration time_from_start
geometry_msgs/PoseWithCovariance pose
""",
    "PredictedObject.msg": """\
uint8 UNKNOWN=0
uint8 CAR=1
uint8 TRUCK=2
uint8 BUS=3
uint8 TRAILER=4
uint8 MOTORCYCLE=5
uint8 BICYCLE=6
uint8 PEDESTRIAN=7
unique_identifier_msgs/UUID object_id
float32 existence_probability
uint8 classification
float32 classification_probability
geometry_msgs/PoseWithCovariance initial_pose
geometry_msgs/TwistWithCovariance initial_twist
geometry_msgs/Vector3 dimensions
PredictedState[] states
""",
    "PredictedObjectArray.msg": """\
std_msgs/Header header
PredictedObject[] objects
""",
}

EXPECTED_TRAFFIC_LIGHT_DECLARATION = """\
std_msgs/Header header
bool valid
float32 confidence
bool red
bool yellow
bool straight_green
bool left_green
string source_class
string detection_id
"""


def _declarations(message_path: Path) -> str:
    return "\n".join(
        line.strip()
        for line in message_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    EXPECTED_PREDICTION_DECLARATIONS.items(),
)
def test_prediction_declarations_remain_stable(filename, expected):
    assert _declarations(PACKAGE_ROOT / "msg" / filename) == (
        expected.strip()
    )


EXPECTED_DYNAMIC_OBJECT_RISK_DECLARATION = """\
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability
float32 x_rel_m
float32 y_rel_m
float32 distance_m
float32 vx_rel_mps
float32 vy_rel_mps
float32 relative_speed_mps
float32 range_rate_mps
float32 longitudinal_closing_mps
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s
DynamicObjectRiskState[] predicted_states
float32 position_uncertainty_m
"""

EXPECTED_CUT_IN_RISK_DECLARATION = """\
uint8 SIDE_NONE=0
uint8 SIDE_LEFT=1
uint8 SIDE_RIGHT=2
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability
float32 route_s_rel_m
float32 lateral_offset_m
float32 corridor_left_width_m
float32 corridor_right_width_m
bool inside_corridor
bool near_boundary
bool adjacent_region
uint8 side
float32 lateral_velocity_mps
float32 lateral_velocity_toward_corridor_mps
bool approaching_corridor
bool longitudinally_relevant
bool predicted_entry_valid
float32 predicted_entry_time_s
float32 predicted_entry_route_s_rel_m
float32 predicted_entry_lateral_offset_m
bool predicted_entry_sustained
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s
bool cut_in_candidate
"""


def test_general_messages_construct_and_keep_prediction_constants():
    assert DynamicObstacleStatus.CLEAR == 0
    assert DynamicObstacleStatus.HAZARD == 2
    assert PlannerStatus() is not None
    assert PredictedState() is not None
    assert PredictedObjectArray() is not None
    assert PredictedObject.CAR == 1
    assert PredictedObject.PEDESTRIAN == 7
    assert TrafficLightStatus() is not None
    assert DynamicObjectRiskState() is not None


def test_dynamic_object_risk_interface_is_stable_and_policy_free():
    """The risk interface exposes metrics only -- never GO / STOP / YIELD."""
    assert _declarations(PACKAGE_ROOT / "msg" / "DynamicObjectRisk.msg") == (
        EXPECTED_DYNAMIC_OBJECT_RISK_DECLARATION.strip()
    )
    assert _declarations(PACKAGE_ROOT / "msg" / "DynamicObjectRiskArray.msg") == (
        "std_msgs/Header header\nDynamicObjectRisk[] objects"
    )
    message = DynamicObjectRisk()
    assert message.ttc_valid is False
    assert message.ttc_s == 0.0
    assert message.cpa_valid is False
    assert list(message.predicted_states) == []
    assert DynamicObjectRiskArray().objects == []
    forbidden = {"go", "stop", "yield", "brake", "risk_score", "decision"}
    fields = {name.lower() for name in DynamicObjectRisk.get_fields_and_field_types()}
    assert fields.isdisjoint(forbidden)


def test_cut_in_risk_interface_is_stable_and_policy_free():
    assert _declarations(PACKAGE_ROOT / "msg" / "CutInRisk.msg") == (
        EXPECTED_CUT_IN_RISK_DECLARATION.strip()
    )
    assert _declarations(PACKAGE_ROOT / "msg" / "CutInRiskArray.msg") == (
        "std_msgs/Header header\nCutInRisk[] risks"
    )
    message = CutInRisk()
    assert message.side == CutInRisk.SIDE_NONE
    assert message.cut_in_candidate is False
    assert message.predicted_entry_valid is False
    assert message.predicted_entry_sustained is False
    assert list(CutInRiskArray().risks) == []
    forbidden = {
        "risk_score", "brake_required", "stop_required", "steer_left",
        "steer_right", "yield", "go",
    }
    fields = {name.lower() for name in CutInRisk.get_fields_and_field_types()}
    assert fields.isdisjoint(forbidden)


EXPECTED_CUT_IN_RESPONSE_DECLARATION = """\
uint8 ACTION_NONE=0
uint8 ACTION_SLOWDOWN=1
uint8 ACTION_HOLD=2
uint8 REASON_NONE=0
uint8 REASON_APPROACHING_ENTRY=1
uint8 REASON_COLLISION_CONFLICT=2
uint8 REASON_SMALL_PREDICTED_CLEARANCE=3
std_msgs/Header header
uint8 action
bool active
uint8 reason
uint16 candidate_count
unique_identifier_msgs/UUID source_object_id
uint8 source_side
bool requested_max_speed_valid
float32 requested_max_speed_mps
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
"""


def test_cut_in_response_is_a_longitudinal_request_not_a_command():
    assert _declarations(PACKAGE_ROOT / "msg" / "CutInResponse.msg") == (
        EXPECTED_CUT_IN_RESPONSE_DECLARATION.strip()
    )
    message = CutInResponse()
    assert message.action == CutInResponse.ACTION_NONE
    assert message.active is False
    assert message.reason == CutInResponse.REASON_NONE
    assert message.requested_max_speed_valid is False
    assert message.requested_max_speed_mps == 0.0
    # A planner-facing request only: no actuator, steering, gear, or path field.
    forbidden = {
        "brake",
        "brake_percentage",
        "throttle",
        "throttle_percentage",
        "steering_angle",
        "steer_left",
        "steer_right",
        "steering",
        "gear",
        "trajectory",
        "path",
        "lane_change",
        "risk_score",
        "acceleration_command",
        "deceleration_command",
    }
    fields = {
        name.lower() for name in CutInResponse.get_fields_and_field_types()
    }
    assert fields.isdisjoint(forbidden)


EXPECTED_ROUNDABOUT_GAP_RISK_DECLARATION = """\
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability
bool relevant_to_conflict
float32 object_map_distance_to_conflict_m
bool object_in_conflict_now
bool object_entry_valid
float32 object_entry_time_s
bool object_exit_valid
float32 object_exit_time_s
bool arrival_delta_valid
float32 arrival_delta_s
bool temporal_gap_valid
float32 temporal_gap_s
bool occupancy_overlap
uint16 predicted_conflict_interval_count
bool later_reentry_detected
bool any_occupancy_overlap
bool minimum_temporal_gap_valid
float32 minimum_temporal_gap_s
float32 prediction_horizon_s
bool prediction_covers_ego_exit
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s
"""

EXPECTED_ROUNDABOUT_GAP_RISK_ARRAY_DECLARATION = """\
std_msgs/Header header
string conflict_zone_id
bool ego_in_conflict_now
bool ego_entry_valid
float32 ego_entry_time_s
bool ego_exit_valid
float32 ego_exit_time_s
float32 ego_route_distance_to_entry_m
float32 ego_route_distance_to_exit_m
float32 ego_speed_mps
uint16 relevant_object_count
RoundaboutGapRisk[] objects
"""


def test_roundabout_gap_risk_interface_is_stable_and_policy_free():
    """Roundabout gap risk exposes conflict-timing facts only -- never GO /
    YIELD / STOP / gap-accepted / safe-to-enter."""
    assert _declarations(PACKAGE_ROOT / "msg" / "RoundaboutGapRisk.msg") == (
        EXPECTED_ROUNDABOUT_GAP_RISK_DECLARATION.strip()
    )
    assert _declarations(PACKAGE_ROOT / "msg" / "RoundaboutGapRiskArray.msg") == (
        EXPECTED_ROUNDABOUT_GAP_RISK_ARRAY_DECLARATION.strip()
    )
    message = RoundaboutGapRisk()
    assert message.relevant_to_conflict is False
    assert message.occupancy_overlap is False
    assert message.temporal_gap_s == 0.0
    # All-interval summary defaults are the conservative "nothing proven" state.
    assert message.predicted_conflict_interval_count == 0
    assert message.later_reentry_detected is False
    assert message.any_occupancy_overlap is False
    assert message.minimum_temporal_gap_valid is False
    assert message.minimum_temporal_gap_s == 0.0
    assert message.prediction_horizon_s == 0.0
    assert message.prediction_covers_ego_exit is False
    assert RoundaboutGapRiskArray().objects == []
    forbidden = {
        "go",
        "stop",
        "yield",
        "yield_required",
        "hold",
        "release",
        "brake",
        "safe_to_enter",
        "gap_accepted",
        "accepted_gap",
        "safe_gap_s",
        "requested_speed",
        "steering",
        "risk_score",
        "decision",
    }
    for message_type in (RoundaboutGapRisk, RoundaboutGapRiskArray):
        fields = {
            name.lower()
            for name in message_type.get_fields_and_field_types()
        }
        assert fields.isdisjoint(forbidden)


EXPECTED_ROUNDABOUT_GAP_RESPONSE_DECLARATION = """\
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
"""


def test_roundabout_gap_response_is_an_advisory_not_a_command():
    assert _declarations(
        PACKAGE_ROOT / "msg" / "RoundaboutGapResponse.msg"
    ) == EXPECTED_ROUNDABOUT_GAP_RESPONSE_DECLARATION.strip()
    message = RoundaboutGapResponse()
    assert message.action == RoundaboutGapResponse.ACTION_RELEASE
    assert message.active is False
    assert message.reason == RoundaboutGapResponse.REASON_NONE
    assert message.source_object_valid is False
    forbidden = {
        "brake", "throttle", "steering", "steering_angle", "cmd_vel",
        "gear", "actuation", "ctrl_cmd", "requested_speed",
        "requested_max_speed_mps", "path", "trajectory",
    }
    fields = {
        name.lower()
        for name in RoundaboutGapResponse.get_fields_and_field_types()
    }
    assert fields.isdisjoint(forbidden)


EXPECTED_HIGHWAY_MERGE_GAP_RISK_DECLARATION = """\
unique_identifier_msgs/UUID object_id
uint8 classification
float32 classification_probability
float32 existence_probability
bool relevant_to_merge
float32 object_route_s_m
float32 object_lateral_offset_m
bool object_in_target_corridor_now
bool predicted_to_enter_target_corridor
bool predicted_corridor_entry_valid
float32 predicted_corridor_entry_time_s
float32 delta_s_now_m
float32 object_longitudinal_speed_mps
float32 relative_longitudinal_speed_mps
bool delta_s_at_merge_valid
float32 delta_s_at_merge_m
bool is_ahead_at_merge
bool is_behind_at_merge
bool is_alongside_at_merge
bool longitudinal_gap_closing
float32 longitudinal_closing_speed_mps
bool time_to_route_coincidence_valid
float32 time_to_route_coincidence_s
bool predicted_min_route_gap_valid
float32 predicted_min_route_gap_m
float32 predicted_min_route_gap_time_s
float32 prediction_horizon_s
bool prediction_covers_merge_time
bool ttc_valid
float32 ttc_s
bool cpa_valid
float32 cpa_time_s
float32 cpa_distance_m
bool predicted_min_separation_valid
float32 predicted_min_separation_m
float32 predicted_min_separation_time_s
"""

EXPECTED_HIGHWAY_MERGE_GAP_RISK_ARRAY_DECLARATION = """\
std_msgs/Header header
string merge_zone_id
string target_lane_sequence_id
string source_lane_sequence_id
float32 merge_zone_entry_route_s_m
float32 merge_reference_route_s_m
float32 ego_route_s_m
float32 ego_longitudinal_speed_mps
float32 ego_route_distance_to_zone_entry_m
float32 ego_route_distance_to_merge_m
bool ego_in_merge_zone_now
bool ego_merge_timing_valid
float32 ego_merge_time_s
uint16 relevant_object_count
bool nearest_leading_valid
unique_identifier_msgs/UUID nearest_leading_object_id
float32 nearest_leading_delta_s_at_merge_m
bool nearest_trailing_valid
unique_identifier_msgs/UUID nearest_trailing_object_id
float32 nearest_trailing_delta_s_at_merge_m
bool merge_gap_valid
float32 merge_gap_m
HighwayMergeGapRisk[] objects
"""


def test_highway_merge_gap_risk_interface_is_stable_and_policy_free():
    """Highway merge gap risk exposes route-relative conflict-geometry facts
    only -- never MERGE / WAIT / lane-change / accepted-gap / safe-to-merge."""
    assert _declarations(
        PACKAGE_ROOT / "msg" / "HighwayMergeGapRisk.msg"
    ) == EXPECTED_HIGHWAY_MERGE_GAP_RISK_DECLARATION.strip()
    assert _declarations(
        PACKAGE_ROOT / "msg" / "HighwayMergeGapRiskArray.msg"
    ) == EXPECTED_HIGHWAY_MERGE_GAP_RISK_ARRAY_DECLARATION.strip()
    message = HighwayMergeGapRisk()
    assert message.relevant_to_merge is False
    assert message.delta_s_at_merge_valid is False
    assert message.is_ahead_at_merge is False
    assert message.is_behind_at_merge is False
    assert message.is_alongside_at_merge is False
    assert message.longitudinal_gap_closing is False
    assert message.prediction_covers_merge_time is False
    assert message.predicted_min_route_gap_m == 0.0
    array = HighwayMergeGapRiskArray()
    assert array.objects == []
    assert array.merge_gap_valid is False
    assert array.nearest_leading_valid is False
    forbidden = {
        "merge_allowed",
        "merge_safe",
        "merge_ready",
        "safe_to_merge",
        "go",
        "wait",
        "yield",
        "yield_required",
        "hold",
        "release",
        "lane_change",
        "change_lane",
        "accelerate",
        "decelerate",
        "accepted_gap",
        "gap_accepted",
        "safe_gap",
        "safe_gap_s",
        "requested_speed",
        "target_speed",
        "brake",
        "throttle",
        "steering",
        "risk_score",
        "decision",
    }
    for message_type in (HighwayMergeGapRisk, HighwayMergeGapRiskArray):
        fields = {
            name.lower()
            for name in message_type.get_fields_and_field_types()
        }
        assert fields.isdisjoint(forbidden)


EXPECTED_HIGHWAY_MERGE_GAP_RESPONSE_DECLARATION = """\
uint8 ACTION_MERGE_READY=0
uint8 ACTION_WAIT=1
uint8 ACTION_HOLD=2
uint8 REASON_NONE=0
uint8 REASON_CLEAR_GAP=1
uint8 REASON_FRONT_GAP=2
uint8 REASON_REAR_GAP=3
uint8 REASON_REAR_CLOSING=4
uint8 REASON_ALONGSIDE=5
uint8 REASON_PREDICTED_ROUTE_CONFLICT=6
uint8 REASON_INSUFFICIENT_PREDICTION=7
uint8 REASON_INVALID_EGO_STATE=8
std_msgs/Header header
uint8 action
bool active
uint8 reason
string merge_zone_id
bool source_object_valid
unique_identifier_msgs/UUID source_object_id
uint16 relevant_object_count
float32 ego_speed_mps
float32 ego_route_distance_to_merge_m
bool ego_merge_timing_valid
float32 ego_merge_time_s
float32 available_distance_m
float32 comfortable_stop_distance_m
bool complete_prediction_coverage
bool front_object_valid
unique_identifier_msgs/UUID front_object_id
float32 front_gap_m
bool front_time_headway_valid
float32 front_time_headway_s
bool rear_object_valid
unique_identifier_msgs/UUID rear_object_id
float32 rear_gap_m
bool rear_time_headway_valid
float32 rear_time_headway_s
bool rear_closing_object_valid
unique_identifier_msgs/UUID rear_closing_object_id
bool rear_closing_time_valid
float32 rear_closing_time_s
bool predicted_route_clearance_valid
float32 minimum_predicted_route_gap_m
"""


def test_highway_merge_gap_response_is_an_advisory_not_a_command():
    """MERGE_READY / WAIT / HOLD are advisory policy states; the message
    carries no actuator, steering, speed, lane-change, or CtrlCmd field."""
    assert _declarations(
        PACKAGE_ROOT / "msg" / "HighwayMergeGapResponse.msg"
    ) == EXPECTED_HIGHWAY_MERGE_GAP_RESPONSE_DECLARATION.strip()
    message = HighwayMergeGapResponse()
    assert message.action == HighwayMergeGapResponse.ACTION_MERGE_READY
    assert message.active is False
    assert message.reason == HighwayMergeGapResponse.REASON_NONE
    assert message.source_object_valid is False
    assert message.complete_prediction_coverage is False
    assert message.front_object_valid is False
    assert message.rear_object_valid is False
    assert message.rear_closing_object_valid is False
    assert message.predicted_route_clearance_valid is False
    forbidden = {
        "brake",
        "brake_percentage",
        "throttle",
        "throttle_percentage",
        "steering",
        "steering_angle",
        "steer_left",
        "steer_right",
        "cmd_vel",
        "ctrl_cmd",
        "gear",
        "actuation",
        "requested_speed",
        "requested_max_speed_mps",
        "target_speed",
        "acceleration_command",
        "deceleration_command",
        "lane_change",
        "lane_change_command",
        "change_lane_now",
        "merge_now",
        "path",
        "trajectory",
        "risk_score",
    }
    fields = {
        name.lower()
        for name in HighwayMergeGapResponse.get_fields_and_field_types()
    }
    assert fields.isdisjoint(forbidden)


def test_traffic_light_status_preserves_composite_aspects():
    """Planner handoff must not collapse composite lights to one enum."""
    assert _declarations(PACKAGE_ROOT / "msg" / "TrafficLightStatus.msg") == (
        EXPECTED_TRAFFIC_LIGHT_DECLARATION.strip()
    )
    message = TrafficLightStatus()
    message.valid = True
    message.red = True
    message.left_green = True
    assert message.red and message.left_green
