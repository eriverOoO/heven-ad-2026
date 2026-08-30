"""Keep simulator-independent HEVEN interface contracts stable."""

from pathlib import Path

import pytest

from ad_interfaces.msg import (
    DynamicObjectRisk,
    DynamicObjectRiskArray,
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
float32 position_uncertainty_m
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
    assert DynamicObjectRiskArray().objects == []
    forbidden = {"go", "stop", "yield", "brake", "risk_score", "decision"}
    fields = {name.lower() for name in DynamicObjectRisk.get_fields_and_field_types()}
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
