# Cut-in evidence for one DynamicObjectRiskArray frame. header.frame_id is the
# configured route-corridor frame (normally map); header.stamp is the source
# risk stamp. An empty risks list is a valid zero-object result.
std_msgs/Header header
CutInRisk[] risks
