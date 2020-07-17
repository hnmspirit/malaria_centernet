import sys
sys.path.append('../cocoapi/PythonAPI')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annType = 'bbox'

gt_fn  = '../data/coco/annotations/instances_test2020.json'
cocoGt = COCO(gt_fn)

dt_fn  = '../ckpt/results_sl1_catspec_e70.json'
cocoDt = cocoGt.loadRes(dt_fn)


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()