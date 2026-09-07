import unittest
from stage_postprocess import greedy_class_agnostic_nms, score_filter

def box(x, score=.5, cls="vehicle"):
    return {"x":x,"y":0.,"length":4.,"width":2.,"yaw":0.,"score":score,"class_name":cls}

class StagePostprocessTest(unittest.TestCase):
    def test_raw_score_nms_flow(self):
        raw=[box(0,.9),box(.1,.8),box(9,.2),box(15,.05),box(20,.04)]
        score=score_filter(raw,.1)
        self.assertEqual(len(raw),5); self.assertEqual(len(score),3)
        self.assertEqual(len(greedy_class_agnostic_nms(score,.5)),2)
    def test_class_agnostic_suppresses_class_conflict(self):
        self.assertEqual(len(greedy_class_agnostic_nms([box(0,.9,"vehicle"),box(.1,.8,"pedestrian")],.5)),1)
    def test_empty(self): self.assertEqual(greedy_class_agnostic_nms([], .5), [])

if __name__ == "__main__": unittest.main()
