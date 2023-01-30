from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

#from lib.external.nms import soft_nms
from lib.models.decode import ctdet_decode, simple_ctdet_decode
from lib.models.utils import flip_tensor
from lib.utils.image import get_affine_transform
from lib.utils.post_process import ctdet_post_process
from lib.utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self):
    super(CtdetDetector, self).__init__()
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      reg_offset=True
      flip_test = False
      K = 100
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if reg_offset else None #if self.opt.reg_offset else None
      if flip_test:#self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      #torch.cuda.synchronize()
      forward_time = time.time()
      dets, x = ctdet_decode(hm, wh, reg=reg, K=K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def simple_process(self, images, return_time=False):
    with torch.no_grad():
      reg_offset=True
      flip_test = False
      K = 100
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if reg_offset else None #if self.opt.reg_offset else None
      if flip_test:#self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      #torch.cuda.synchronize()
      forward_time = time.time()
      x = simple_ctdet_decode(hm, wh, reg=reg, K=K)

    if return_time:
      return x, forward_time
    else:
      return x

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = 170
    #print("DETS AFTER RESHAPE", dets.shape)
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)#self.opt.num_classes)
    #print("DETS POST PROCESS", dets)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    #print("DETS[0]", dets[0])
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    nms = False
    #print("DETECTIONS", detections)
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      #if len(self.scales) > 1 or nms:#self.opt.nms:
      #   soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    #print("INI RESULTS FROM MERGE OUTPUT", results[4])
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= 4#self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > 0.1:#self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4],
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] >= 0.48:#self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)