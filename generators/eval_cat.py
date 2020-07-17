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
params = cocoEval.params
params.iouThrs = [0.2,]
params.maxDets = [100,]
params.areaRng = [[0 ** 2, 1e5 ** 2],]
params.areaRngLbl = ['all',]

cocoEval.evaluate()
cocoEval.accumulate()

precision = cocoEval.eval['precision'].squeeze().mean(axis=0)
print('precision: {:.3f}, {:.3f}; {:.3f}'.format(*precision, precision.mean()))

recall = cocoEval.eval['recall'].squeeze()
print('recall   : {:.3f}, {:.3f}; {:.3f}'.format(*recall, recall.mean()))