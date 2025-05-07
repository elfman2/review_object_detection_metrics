################################################################################
# Tests performed in the PASCAL VOC metrics
################################################################################

import unittest
from math import isclose

import odmetrics.utils.converter as converter
from odmetrics.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from odmetrics.utils.enumerators import BBType, MethodAveragePrecision
import numpy as np
class TestPascalVOC(unittest.TestCase):

    def test_case_1(self):
        gts_dir = 'tests/test_case_1/gts'
        dets_dir = 'tests/test_case_1/dets'

        gts = converter.text2bb(gts_dir, BBType.GROUND_TRUTH)
        dets = converter.text2bb(dets_dir, BBType.DETECTED)

        self.assertTrue (len(gts) > 0)
        self.assertTrue (len(dets) > 0)

        testing_ious = [0.1, 0.3, 0.5, 0.75]

        # ELEVEN_POINT_INTERPOLATION
        expected_APs = {'object': {0.1: 0.3333333333, 0.3: 0.2683982683, 0.5: 0.0303030303, 0.75: 0.0}}
        for idx, iou in enumerate(testing_ious):
            results_dict = get_pascalvoc_metrics(
                gts, dets, iou_threshold=iou, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
            results = results_dict['per_class']
            for c, res in results.items():
                self.assertTrue (isclose(expected_APs[c][iou], res['AP']))

        # EVERY_POINT_INTERPOLATION
        expected_APs = {'object': {0.1: 0.3371980676, 0.3: 0.2456866804, 0.5: 0.0222222222, 0.75: 0.0}}
        for idx, iou in enumerate(testing_ious):
            results_dict = get_pascalvoc_metrics(
                gts, dets, iou_threshold=iou, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)
            results = results_dict['per_class']
            for c, res in results.items():
                self.assertTrue (isclose(expected_APs[c][iou], res['AP']))

    def test_case_2(self):
        gts_dir = 'tests/test_case_1/gts'
        dets_dir = 'tests/test_case_1/dets'

        gts = converter.text2bb(gts_dir, BBType.GROUND_TRUTH)
        dets = converter.text2bb(dets_dir, BBType.DETECTED)
        self.assertTrue (len(gts) > 0)
        self.assertTrue (len(dets) > 0)

        testing_ious = np.linspace(0.1,0.55,10)
        eps_list = np.logspace(-3,1,10) # sample epsilon logarithmically from 1e-3 to 1e+1
        for idx, iou in enumerate(testing_ious):
            for eps in eps_list:
                previous_map = 1.
                results_dict = get_pascalvoc_metrics(gts, dets, 
                                                        iou_threshold=iou, 
                                                        method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION,
                                                        eps = eps)
#                print(iou,eps,results_dict['mAP'])
                self.assertTrue(results_dict['mAP']<=previous_map)
                previous_map = results_dict['mAP']
if __name__ == '__main__':
    unittest.main()