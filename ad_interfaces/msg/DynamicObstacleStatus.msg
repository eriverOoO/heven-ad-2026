std_msgs/Header header

uint8 CLEAR=0
uint8 CANDIDATE=1
uint8 HAZARD=2
uint8 STALE=3

uint8 state
bool valid
bool stop_required
uint32 candidate_count
float32 max_confidence
float32 selected_foot_x_normalized
float32 selected_foot_y_normalized
