std_msgs/Header header
bool has_device_stamp
builtin_interfaces/Time device_stamp
int8 ctrl_mode
int8 gear
# ROS-facing longitudinal speed is in m/s.
float32 signed_velocity
int32 map_data_id
float32 accel
float32 brake
# Vehicle dimensions are in metres.
geometry_msgs/Vector3 size
float32 overhang
float32 wheelbase
float32 rear_overhang
# Map-frame position is in metres.
geometry_msgs/Vector3 position
# Roll, pitch, and yaw are in radians.
geometry_msgs/Vector3 rpy
# Linear velocity is in m/s.
geometry_msgs/Vector3 velocity
# Angular velocity is in rad/s.
geometry_msgs/Vector3 angular_velocity
# Linear acceleration is in m/s^2.
geometry_msgs/Vector3 acceleration
# Steering angle is in radians.
float32 steering
string link_id
bool has_tire_metrics
float32 tire_lateral_force_fl
float32 tire_lateral_force_fr
float32 tire_lateral_force_rl
float32 tire_lateral_force_rr
float32 side_slip_angle_fl
float32 side_slip_angle_fr
float32 side_slip_angle_rl
float32 side_slip_angle_rr
float32 tire_cornering_stiffness_fl
float32 tire_cornering_stiffness_fr
float32 tire_cornering_stiffness_rl
float32 tire_cornering_stiffness_rr
