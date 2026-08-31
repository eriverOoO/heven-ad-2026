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
    assert RoundaboutGapRiskArray().objects == []
    forbidden = {
        "go",
        "stop",
        "yield",
        "yield_required",
        "hold",
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
