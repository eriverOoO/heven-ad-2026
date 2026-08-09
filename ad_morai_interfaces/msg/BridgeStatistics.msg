std_msgs/Header header
string stream
uint64 packets
uint64 bytes
uint64 malformed
uint64 dropped
uint64 bind_errors
uint64 source_selected
uint64 arrival_fallback
uint64 source_rejected
# Count of repeated accepted source stamps, not confirmed retransmissions.
uint64 duplicates
uint64 stamp_regressions
float64 packet_rate_hz
float64 byte_rate_bps
float64 last_packet_age_sec
