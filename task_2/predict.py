from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import time
import csv
import io
import requests
import random
import torch

import xml.etree.cElementTree as ET

from lib.detectors.detector_factory import detector_factory

Detector = detector_factory["ctdet"]
detector = Detector()

def bb_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(float(boxA[0]), float(boxB[0]))
  yA = max(float(boxA[1]), float(boxB[1]))
  xB = min(float(boxA[2]), float(boxB[2]))
  yB = min(float(boxA[3]), float(boxB[3]))
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (float(boxA[2]) - float(boxA[0]) + 1) * (float(boxA[3]) - float(boxA[1]) + 1)
  boxBArea = (float(boxB[2]) - float(boxB[0]) + 1) * (float(boxB[3]) - float(boxB[1]) + 1)
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
  # return the intersection over union value
  return iou

n_gpu = torch.cuda.device_count()

def set_seed():
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)


def predict(img_file):
  img = cv2.imread(img_file)
  filename, ext = os.path.splitext(img_file)

  prev_height, prev_width, _ = img.shape

  set_seed()

  classes = ['ponds']

  ret = detector.run(img_file)
  results = ret["results"]

  list_dets = []

  for j in range(1, len(results)+1):
    for bbox in results[j]:
      if bbox[4] >= 0.3:
        list_dets.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], classes[j-1], bbox[4]]))

  takedown_index = {}
  for a in range(len(list_dets)):
  	for b in range(len(list_dets)):
  		iou = bb_intersection_over_union(list_dets[a][:4], list_dets[b][:4])
  		if iou >= 0.5 and list_dets[a][4] != list_dets[b][4]:
  			if list_dets[a][5] > list_dets[b][5]:
  				if b not in takedown_index:
  					takedown_index[list_dets[b][0]] = list_dets[b][4]
  				else:
  					pass
  			else:
  				if a not in takedown_index:
  					takedown_index[list_dets[a][0]] = list_dets[a][4]
  				else:
  					pass

  list_dets = np.array(list_dets)
  idx = []
  for i in takedown_index:
  	idx_class = (np.where(list_dets == i))
  	idx.append(idx_class[0][0])

  list_dets = np.delete(list_dets, idx, axis=0)

  container = []

  jumlah_kolam = len(list_dets)

  for i in range(len(list_dets)):
     centre_coordinate = (int((float(list_dets[i][2]) + float(list_dets[i][0]))/2), int((float(list_dets[i][3]) + float(list_dets[i][1]))/2))
     img = cv2.circle(img, centre_coordinate, 6, (0, 0, 255), -1)

  cv2.imwrite(filename+"_dot_seed"+".png", img)

  return {"image_visualization":filename+"_dot_seed"+".png", 'jumlah_kolam':jumlah_kolam} 
