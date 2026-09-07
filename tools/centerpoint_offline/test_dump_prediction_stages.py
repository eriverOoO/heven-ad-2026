import unittest
import numpy as np
from dump_prediction_stages import STAGE_NAMES, serial

class DumpPredictionStagesTest(unittest.TestCase):
    def test_stage_provenance_order_is_stable(self):
        self.assertEqual(STAGE_NAMES, ('raw_decoded','score_filtered','post_nms','ros_final'))
    def test_serial_preserves_candidate_class_and_geometry(self):
        rows=serial(np.array([[1,2,3,4,5,6,.7]],dtype=float),np.array([.8]),np.array([1]),('vehicle','pedestrian'))
        self.assertEqual(rows[0]['candidate_id'],0); self.assertEqual(rows[0]['class_name'],'pedestrian'); self.assertEqual(rows[0]['box_lidar'][3],4.0)

if __name__=='__main__': unittest.main()
