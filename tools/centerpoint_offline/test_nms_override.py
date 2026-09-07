import unittest
from nms_override import effective_nms_threshold

class NmsOverrideTest(unittest.TestCase):
    def test_default_preserves_historical_threshold(self): self.assertEqual(effective_nms_threshold(.7,None),.7)
    def test_override_propagates(self):
        self.assertEqual(effective_nms_threshold(.7,.3),.3); self.assertEqual(effective_nms_threshold(.7,.1),.1)
    def test_candidate_requests_are_independent(self):
        self.assertEqual([effective_nms_threshold(.7,x) for x in (.7,.5,.3,.2,.1)],[.7,.5,.3,.2,.1])
    def test_rejects_invalid(self):
        with self.assertRaises(ValueError): effective_nms_threshold(.7,1.1)

if __name__=='__main__': unittest.main()
