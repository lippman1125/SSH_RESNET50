# ------------------------------------------
# SSH: Single Stage Headless Face Detector
# Demo
# by Mahyar Najibi
# ------------------------------------------

from __future__ import print_function
from SSH.test import detect_list
from argparse import ArgumentParser
import os
import sys
import cv2
import numpy as np
import argparse
from utils.get_config import cfg_from_file, cfg, cfg_print
from nms.nms_wrapper import nms
from utils.test_utils import visusalize_detections
import caffe


# def parser():
#     parser = ArgumentParser('SSH Demo!')
#     parser.add_argument('--im',dest='im_path',help='Path to the image',
#                         default='data/demo/demo.jpg',type=str)
#     parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
#                         default=0,type=int)
#     parser.add_argument('--proto',dest='prototxt',help='SSH caffe test prototxt',
#                         default='SSH/models/test_ssh.prototxt',type=str)
#     parser.add_argument('--model',dest='model',help='SSH trained caffemodel',
#                         default='data/SSH_models/SSH.caffemodel',type=str)
#     parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
#                         default='data/demo',type=str)
#     parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
#                         default='SSH/configs/wider_pyramid.yml',type=str)
#     return parser.parse_args()
#
# if __name__ == "__main__":
#
#     # Parse arguments
#     args = parser()
#
#     # Load the external config
#     if args.cfg is not None:
#         cfg_from_file(args.cfg)
#     # Print config file
#     cfg_print(cfg)
#
#     # Loading the network
#     cfg.GPU_ID = args.gpu_id
#     caffe.set_mode_gpu()
#     caffe.set_device(args.gpu_id)
#     # caffe.set_mode_cpu()
#     assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
#     assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'
#
#     print('Loading the network...', end="")
#     net = caffe.Net(args.prototxt, args.model, caffe.TEST)
#     net.name = 'SSH'
#     print('Done!')
#
#     # Read image
#     assert os.path.isfile(args.im_path),'Please provide a path to an existing image!'
#     pyramid = True if len(cfg.TEST.SCALES)>1 else False
#
#     # Perform detection
#     cls_dets,_ = detect(net,args.im_path,visualization_folder=args.out_path,visualize=True,pyramid=pyramid)


def demo_test(net, im, pyramid):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    probs, boxes = detect_list(net, im, pyramid=pyramid)

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.3
    # for cls_ind, cls in enumerate(CLASSES[1:]):
    #     cls_ind += 1 # because we skipped background
    #     if cls_name == cls:
    #         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #         cls_scores = scores[:, cls_ind]
    #         dets = np.hstack((cls_boxes,
    #                           cls_scores[:, np.newaxis])).astype(np.float32)
    #         keep = nms(dets, NMS_THRESH)
    #         dets = dets[keep, :]
    #         # vis_detections(im, cls, dets, thresh=CONF_THRESH)
    #         inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    #         dets = dets[inds]

    inds = np.where(probs[:, 0] > CONF_THRESH)[0]
    probs = probs[inds, 0]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, probs[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets

def parse_args(argv):
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--image_dir', type=str, help='images directory')
    parser.add_argument('--file_list', type=str, help='images list')
    parser.add_argument('--file_result', type=str, help='result of detection')
    parser.add_argument('--output_dir',dest='output_dir',help='Output path for saving the pictures',default='data/output_dir',type=str)

    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    cfg_file = 'SSH/configs/wider.yml'
    cfg_from_file(cfg_file)
    # Print config file
    cfg_print(cfg)

    # prototxt = 'SSH/models/test_ssh_resnet50_bn.prototxt'
    # caffemodel = 'data/SSH_models/SSH_resnet50_iter_84000.caffemodel'

    prototxt = 'SSH/models/test_ssh_pvanet.prototxt'
    caffemodel = 'data/SSH_models/SSH_pvanet_iter_84000.caffemodel'


    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    image_dir = args.image_dir
    file_list = args.file_list
    file_result = args.file_result
    output_dir = args.output_dir

    if not os.path.exists(image_dir):
        print("image_dir: {} does not exist".format(image_dir))
        exit()
    if not os.path.exists(file_list):
        print("file_list: {} does not exist".format(file_list))
        exit()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    print('Loading the network...', end="")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = "SSH"
    print('Done!')

    pyramid = True if len(cfg.TEST.SCALES) > 1 else False

    imgs_path_fd = open(file_list, "r")
    imgs_path = imgs_path_fd.readlines()
    imgs_path_fd.close()

    count = 0
    _str = ""

    for img_path in imgs_path:
        full_path = os.path.join(image_dir, img_path.strip("\n") + ".jpg")
        print(full_path)
        img = cv2.imread(full_path)
        dets = demo_test(net, img, pyramid)

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        s  = dets[:, 4]

        img_basename = os.path.basename(full_path)
        plt_name = img_path.replace("/", "_").strip("\n")
        visusalize_detections(img, dets, plt_name=plt_name, visualization_folder=output_dir)

        str_name = img_path.strip("\n") + "\n"
        str_box = ""
        count = 0
        for i in range(dets.shape[0]):
            str_box += str(x1[i]) + " " \
                       + str(y1[i]) + " " \
                       + str(x2[i] - x1[i]) + " " \
                       + str(y2[i] - y1[i]) + " " \
                       + str(s[i]) + "\n"
            count += 1

        _str += str_name
        _str += str(count) + "\n"
        _str += str_box

        print(str_name)
        print(count)
        print(str_box)

    d_ret_fd = open(file_result, "w")
    d_ret_fd.writelines(_str)
    d_ret_fd.close()