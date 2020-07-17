import sys
sys.path.append('../cocoapi/PythonAPI')

import os, glob
from os.path import isdir, isfile, join, basename  as bn
import json
import numpy as np
import cv2

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annType = 'bbox'


gt_fn  = '../data/coco/annotations/instances_test2020.json'
cocoGt = COCO(gt_fn)

dt_fn  = '../ckpt/results_sl1_catspec_e70.json'
cocoDt = cocoGt.loadRes(dt_fn)

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
leparams = cocoEval.params
leparams.iouThrs = [0.2,]
leparams.maxDets = [100,]
leparams.areaRng = [[0 ** 2, 1e5 ** 2],]
leparams.areaRngLbl = ['all',]
cocoEval.evaluate()

with open(gt_fn) as f:
    gt_coco = json.load(f)
with open(dt_fn) as f:
    dt_coco = json.load(f)

img_root = '../data/coco/images/test2020'
dst_root = 'oracle'
os.makedirs(dst_root, exist_ok=True)


RED = (0,0,255)
GREEN = (0,255,0)

BLUE = (255,0,0)
BLUED = (100,0,0)

ORANGE = (0,128,255)
ORANGED = (0,30,100)

WHITE = (255,255,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

cats = [1, 2]

nimgs = len(gt_coco['images'])

for img_idx, ima in enumerate(gt_coco['images']):
#     if img_idx > 3:
#         break
    imid = ima['id']
    if img_idx % 10 == 0:
        print('idx: {}: {}'.format(img_idx, imid))
    img_fn = join(img_root, ima['file_name'])
    img = cv2.imread(img_fn, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img0 = img.copy()

    save_name = bn(ima['file_name']).split('.')[0]
    fn_flag = False
    fp_flag = False

    gta = [a for a in gt_coco['annotations'] if a['image_id'] == imid]
    dta = [a for a in dt_coco if a['image_id'] == imid]

    for ct_idx, ctid in enumerate(cats):
        gta1 = [a for a in gta if a['category_id'] == ctid]
        evalM = cocoEval.evalImgs[ct_idx * nimgs + img_idx]
        gtm = evalM['gtMatches'][0]

        for a_idx, ann in enumerate(gta1):
            if gtm[a_idx] == 0:
               color = RED
               fn_flag = True
            else:
               color = GREEN
            x,y,w,h = ann['bbox']
            x,y,x2,y2 = int(x), int(y), int(x+w), int(y+h)
            cv2.rectangle(img, (x,y),(x2,y2), color, -1)
    img = cv2.addWeighted(img0, 0.5, img, 0.5, 0)

    for ct_idx, ctid in enumerate(cats):
        dta1 = [a for a in dta if a['category_id'] == ctid]
        evalM = cocoEval.evalImgs[ct_idx * nimgs + img_idx]
        dtm = evalM['dtMatches'][0]
        dts = evalM['dtScores']

        for a_idx, ann in enumerate(dta1):
            if dtm[a_idx] == 0:
                if dts[a_idx] < 0.3:
                   continue
                color = ORANGE
                colord = ORANGED
                fp_flag = True
            else:
                color = BLUE
                colord = BLUED
            x,y,w,h = ann['bbox']
            x,y,x2,y2 = int(x), int(y), int(x+w), int(y+h)
            cv2.rectangle(img, (x,y),(x2,y2), color, 2)

            text = '{:.2f}'.format(dts[a_idx])
            cv2.rectangle(img, (x-1,y-14),(x+30,y), colord, -1)
            cv2.putText(img, text, (x,y-2), FONT, 0.4, WHITE, 1)

    if fn_flag:
        save_name += '_fn'
    if fp_flag:
        save_name += '_fp'
    savepath = join(dst_root, save_name + '.jpg')
    cv2.imwrite(savepath, img)