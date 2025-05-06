import json
from math import isclose
import unittest

from odmetrics.bounding_box import BBFormat, BBType, BoundingBox
from odmetrics.evaluators.coco_evaluator import get_coco_summary
from odmetrics.utils.converter import coco2bb

class TestCoco(unittest.TestCase):
    def test_coco(self):
        # Load coco samples
        gts = coco2bb('tests/test_coco_eval/gts', BBType.GROUND_TRUTH)
        dts = coco2bb('tests/test_coco_eval/dets', BBType.DETECTED)

        res = get_coco_summary(gts, dts)

        # Compare results to those obtained with coco's official implementation
        tol = 1e-6

        self.assertTrue( abs(res["AP"] - 0.503647) < tol)
        self.assertTrue( abs(res["AP50"] - 0.696973) < tol)
        self.assertTrue( abs(res["AP75"] - 0.571667) < tol)
        self.assertTrue( abs(res["APsmall"] - 0.593252) < tol)
        self.assertTrue( abs(res["APmedium"] - 0.557991) < tol)
        self.assertTrue( abs(res["APlarge"] - 0.489363) < tol)
        self.assertTrue( abs(res["AR1"] - 0.386813) < tol)
        self.assertTrue( abs(res["AR10"] - 0.593680) < tol)
        self.assertTrue( abs(res["AR100"] - 0.595353) < tol)
        self.assertTrue( abs(res["ARsmall"] - 0.654764) < tol)
        self.assertTrue( abs(res["ARmedium"] - 0.603130) < tol)
        self.assertTrue( abs(res["ARlarge"] - 0.553744) < tol)
