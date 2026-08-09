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
# Twist is expressed in the PredictedObjectArray header frame.
geometry_msgs/TwistWithCovariance initial_twist
geometry_msgs/Vector3 dimensions
PredictedState[] states
