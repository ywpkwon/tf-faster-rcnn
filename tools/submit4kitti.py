#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
from tqdm import tqdm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__',  # always index 0
           'Car', 'Van', 'Truck',
           'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
           'Misc', 'DontCare')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_160000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}

DATASETS = {'pascal_voc': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
            'kitti': ('kitti_2012_train_diff_',)}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    template = '{} {} {} {} {} {} {:.4f}\n'
    lines = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        final_bboxes = dets[inds, :4]
        final_scores = dets[inds, -1]

        for b, s in zip(final_bboxes, final_scores):
            bbox_str = ' '.join(['{:.4f}'.format(f) for f in b])
            bbox_3d_str = ' '.join(['0.000'] * 7)       # dummy
            lines.append(template.format(cls, 0, 0, -10, bbox_str, bbox_3d_str, s))
    return ''.join(lines)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', choices='DATASETS.keys()', default='kitti')
    parser.add_argument("--kitti_dir", default="/mnt/data/data/kitti/", action="store", help="")
    parser.add_argument("--outdir", default="submit", action="store", help="a directory name to output submission")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError

    net.create_architecture("TEST", len(CLASSES), tag='default',
                            anchor_scales=[4, 8, 16, 32],
                            anchor_ratios=[0.5, 1, 2])

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    if not os.path.isdir(args.outdir):   os.mkdir(args.outdir)

    for split in ['train', 'val', 'test']:

        # --- set image directory
        targetf = os.path.join(args.kitti_dir, 'ImageSets', '{}.txt'.format(split))
        if split == 'test':  img_dir = os.path.join(args.kitti_dir, 'testing')
        else:                img_dir = os.path.join(args.kitti_dir, 'training')
        imgf = os.path.join(args.kitti_dir, img_dir, 'image_2', '{}.png')

        # --- read split
        with open(targetf, 'r') as fp:
            lines = fp.readlines()
            names = [line.strip() for line in lines]

        # --- set output directory
        outdir = os.path.join(args.outdir, split)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        else:
            raise IOError(('The directory "{:s}" already exists').format(outdir))

        # --- do the job
        for name in tqdm(names):
            with open(os.path.join(outdir, name + '.txt'), 'w') as fp:
                im_file = imgf.format(name)
                im = cv2.imread(im_file)
                lines = demo(sess, net, im)
                fp.write(lines)
