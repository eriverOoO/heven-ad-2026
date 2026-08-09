STREAMS = {
    "ego_status": (1911, "/ad/dev/vehicle/ego_status", "map", "ego"),
    "objects": (7505, "/ad/dev/objects", "map", "objects"),
    "lidar2d": (9301, "/ad/dev/lidar2d/scan", "lidar2d_link", "lidar2d"),
    "traffic_light_status": (
        7502, "/ad/dev/traffic_light/status", "map", "traffic_light"
    ),
    "intersection_status": (
        9102, "/ad/dev/intersection/status", "map", "intersection"
    ),
    "npc_collisions": (
        9108, "/ad/dev/npc/collisions", "map", "npc_collisions"
    ),
}


OUTPUTS = {
    "ego_ghost": 9095,
    "traffic_light": 7607,
    "intersection": 9132,
    "sensor_control": 9103,
    "lamp_control": 9097,
    "scenario_load": 9099,
    "save_sensor_data": 9105,
    "multi_ego": 7604,
    "npc_ghost": 9101,
}
