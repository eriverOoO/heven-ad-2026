from ad_morai_interfaces_dev.msg import (
    EgoGhostCmd,
    IntersectionControl,
    Lamps,
    MultiEgoSetting,
    MultiEgoVehicle,
    NpcGhostCmd,
    NpcGhostInfo,
    SaveSensorData,
    ScenarioLoad,
    SensorPosControl,
    TrafficLightControl,
)

from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge


def _bridge(senders):
    bridge = object.__new__(AdMoraiDevBridge)
    bridge._senders = senders
    bridge.errors = []
    bridge.get_logger = lambda: type(
        "Logger",
        (),
        {"error": lambda _self, message: bridge.errors.append(message)},
    )()
    return bridge


def _messages():
    ego = EgoGhostCmd()
    ego.position.x, ego.position.y, ego.position.z = (1.0, 2.0, 3.0)
    ego.rpy.x, ego.rpy.y, ego.rpy.z = (4.0, 5.0, 6.0)
    ego.velocity, ego.steering = (7.0, 8.0)

    traffic = TrafficLightControl()
    traffic.traffic_light_index = "C119BS010001"
    traffic.traffic_light_status = 16

    intersection = IntersectionControl()
    intersection.intersection_index = 2
    intersection.intersection_status = 4
    intersection.intersection_time = 3.5

    sensor = SensorPosControl()
    sensor.sensor_index = 3
    sensor.position.x, sensor.position.y, sensor.position.z = (11.0, 12.0, 13.0)
    sensor.rpy.x, sensor.rpy.y, sensor.rpy.z = (14.0, 15.0, 16.0)

    lamps = Lamps()
    lamps.turn_signal = 1
    lamps.emergency_signal = 2

    scenario = ScenarioLoad()
    scenario.filename = "scenario.json"
    scenario.delete_all = True
    scenario.network = False
    scenario.ego = True
    scenario.npc = False
    scenario.pedestrian = True
    scenario.object = False
    scenario.pause = True

    save = SaveSensorData()
    save.custom = True
    save.file_name = "capture"
    save.file_dir = "/tmp/data"

    vehicle = MultiEgoVehicle()
    vehicle.ego_index = 5
    vehicle.position.x, vehicle.position.y, vehicle.position.z = (21.0, 22.0, 23.0)
    vehicle.rpy.x, vehicle.rpy.y, vehicle.rpy.z = (24.0, 25.0, 26.0)
    vehicle.velocity = 27.0
    vehicle.gear = 4
    vehicle.ctrl_mode = 2
    multi = MultiEgoSetting()
    multi.camera_index = 6
    multi.vehicles = [vehicle]

    ghost_info = NpcGhostInfo()
    ghost_info.unique_id = 7
    ghost_info.car_name = "2016_Kia_Niro"
    ghost_info.position.x, ghost_info.position.y, ghost_info.position.z = (
        31.0,
        32.0,
        33.0,
    )
    ghost_info.rpy.x, ghost_info.rpy.y, ghost_info.rpy.z = (34.0, 35.0, 36.0)
    npc = NpcGhostCmd()
    npc.vehicles = [ghost_info]

    return {
        "ego_ghost": ("_on_ego_ghost", ego),
        "traffic_light": ("_on_traffic_light_control", traffic),
        "intersection": ("_on_intersection_control", intersection),
        "sensor_control": ("_on_sensor_pos_control", sensor),
        "lamp_control": ("_on_lamps", lamps),
        "scenario_load": ("_on_scenario_load", scenario),
        "save_sensor_data": ("_on_save_sensor_data", save),
        "multi_ego": ("_on_multi_ego", multi),
        "npc_ghost": ("_on_npc_ghost", npc),
    }


def test_typed_callbacks_map_every_field_through_shared_encoder(monkeypatch):
    captured = []
    sent = []

    def encode(kind, values):
        captured.append((kind, values))
        return f"encoded:{kind}".encode()

    from ad_morai_bridge_dev.bridge import node as node_module

    monkeypatch.setattr(node_module, "encode_simulator_command", encode)
    sender = type("Sender", (), {"send": lambda _self, packet: sent.append(packet)})()
    bridge = _bridge({kind: sender for kind in _messages()})

    for kind, (callback, message) in _messages().items():
        getattr(bridge, callback)(message)
        assert captured[-1][0] == kind

    assert len(captured) == len(sent) == 9
    values = dict(captured)
    assert values["ego_ghost"] == {
        "pose_x": 1.0,
        "pose_y": 2.0,
        "pose_z": 3.0,
        "roll": 4.0,
        "pitch": 5.0,
        "yaw": 6.0,
        "velocity": 7.0,
        "steering": 8.0,
    }
    assert values["traffic_light"] == {
        "traffic_light_index": "C119BS010001",
        "traffic_light_status": 16,
    }
    assert values["intersection"] == {
        "intersection_index": 2,
        "intersection_status": 4,
        "intersection_time": 3.5,
    }
    assert values["sensor_control"]["yaw"] == 16.0
    assert values["lamp_control"] == {"turn_signal": 1, "emergency_signal": 2}
    assert values["scenario_load"]["filename"] == "scenario.json"
    assert values["scenario_load"]["pause"] is True
    assert values["save_sensor_data"] == {
        "custom": True,
        "file_name": "capture",
        "file_dir": "/tmp/data",
    }
    assert values["multi_ego"]["camera_index"] == 6
    assert values["multi_ego"]["vehicles"][0]["ego_index"] == 5
    assert values["multi_ego"]["vehicles"][0]["position_x"] == 21.0
    assert values["multi_ego"]["vehicles"][0]["position_y"] == 22.0
    assert values["multi_ego"]["vehicles"][0]["position_z"] == 23.0
    assert values["multi_ego"]["vehicles"][0]["yaw"] == 26.0
    assert values["npc_ghost"]["vehicles"][0]["unique_id"] == 7
    assert values["npc_ghost"]["vehicles"][0]["yaw"] == 36.0


def test_shared_send_path_rejects_disabled_sender_without_encoding(monkeypatch):
    encoded = []
    from ad_morai_bridge_dev.bridge import node as node_module

    monkeypatch.setattr(
        node_module,
        "encode_simulator_command",
        lambda kind, values: encoded.append((kind, values)),
    )
    bridge = _bridge({})

    bridge._send_simulator_command("scenario_load", {"filename": "x"})

    assert encoded == []
    assert bridge.errors and "no enabled output" in bridge.errors[0]
