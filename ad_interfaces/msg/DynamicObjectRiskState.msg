# One discrete future object-centroid state carried by DynamicObjectRisk.
# Coordinates are relative to the constant-velocity ego rollout at the same
# time_from_start and are expressed in the parent array's base_link axes at the
# source stamp (+x forward, +y left). No extrapolation is added here.
builtin_interfaces/Duration time_from_start
float32 x_rel_m
float32 y_rel_m
