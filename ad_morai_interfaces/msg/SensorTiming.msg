# Arrival time is always structurally valid and remains the audit header.
std_msgs/Header header
# Audit event label; GPS distinguishes gps/rmc from gps/fix.
string stream

# Signed raw fields preserve even a structurally invalid MORAI source pair.
bool has_source_stamp
int64 source_sec
int64 source_nanosec

# The candidate header selected by the policy, including for a dropped sample.
builtin_interfaces/Time selected_stamp
bool source_valid
bool source_selected
bool source_rejected
# True means the same accepted source stamp repeated within this stream.  MORAI
# does not define this as retransmission, so it is audit-only and never by itself
# suppresses normalized publication.
bool duplicate
bool stamp_regression
bool normalized_published
