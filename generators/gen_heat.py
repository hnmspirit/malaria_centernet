import sys
sys.path.append('../cocoapi/PythonAPI')
sys.path.append('../lib/utils')

import os, glob
from os.path import isdir, isfile, join, basename  as bn
import json
import numpy as np
import cv2

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from image import gaussian_radius, draw_gaussian

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
dst_root2 = 'heat'
os.makedirs(dst_root2, exist_ok=True)


# RED = (0,0,255)
# GREEN = (0,255,0)

# BLUE = (255,0,0)
# BLUED = (100,0,0)
# ORANGE = (0,128,255)
# ORANGED = (0,30,100)

# WHITE = (255,255,255)
# FONT = cv2.FONT_HERSHEY_SIMPLEX

cats = [1, 2]

nimgs = len(gt_coco['images'])

for img_idx, ima in enumerate(gt_coco['images']):
    # if img_idx > 30:
    #     break
    imid = ima['id']
    if img_idx % 10 == 0:
        print('idx: {}: {}'.format(img_idx, imid))
    img_fn = join(img_root, ima['file_name'])
    heatmap = np.zeros((600,600), np.float32)
    dta = [a for a in dt_coco if a['image_id'] == imid]

    for ct_idx, ctid in enumerate(cats):
        dta1 = [a for a in dta if a['category_id'] == ctid]
        evalM = cocoEval.evalImgs[ct_idx * nimgs + img_idx]
        dtm = evalM['dtMatches'][0]
        dts = evalM['dtScores']

        for a_idx, ann in enumerate(dta1):
            if dtm[a_idx] == 0:
                if dts[a_idx] < 0.2:
                   continue
            x,y,w,h = ann['bbox']
            center = [int(x+w/2), int(y+h/2)]
            w, h = int(w), int(h)
            radius = gaussian_radius((h,w), iou=0.3)
            radius = int(radius)

            draw_gaussian(heatmap, center, radius, k=min(ann['score']*1.05,1))

    # img = np.uint8(heatmap * 255)

    save_name = bn(ima['file_name']).split('.')[0]
    savepath = join(dst_root2, save_name + '.jpg')

    fig = plt.figure(figsize=(1,1));
    plt.imshow(heatmap, cmap='viridis');
    plt.xticks([]); plt.yticks([]); plt.axis('equal')
    plt.savefig(savepath, dpi=600, bbox_inches='tight', pad_inches=0);
    plt.close()