# Planner-facing per-object relative-motion / risk metrics for one prediction
# frame. Produced by the Dynamic Object Risk Interface from a canonical
# ad_interfaces/PredictedObjectArray plus the ego state.
#
# header.frame_id is the ego body frame (base_link); header.stamp is the
# source PredictedObjectArray stamp. An empty objects list with a valid
# header is a valid "no dynamic objects" result.
std_msgs/Header header
DynamicObjectRisk[] objects
