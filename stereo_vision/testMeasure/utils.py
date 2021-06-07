from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import stereo_vision.testMeasure._init_paths ##这里还要改回来
# import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from xml.dom import minidom
import json
import math

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

from xml.dom import minidom


#########################目标检测部分###########################


CLASSES = ('__background__','hole','target')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_55000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(54.72, 36.48))
    ax.imshow(im, aspect='equal')
    coordinates_list = list()
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        coordinates_list.append([bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]])
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=9)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=30, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    return coordinates_list


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    classes_list = list()
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        classes_list.append(vis_detections(im, cls, dets, thresh=CONF_THRESH))
    return classes_list

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

def get_img_boxes(flag):
    check = False  # 用来返回检测到的数据是否合乎标准
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    tfmodel = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure","tf-faster-rcnn/output/res101/voc_2007_trainval/"
                                                                                "default/res101_faster_rcnn_iter_110000.ckpt")
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\n请检查output路径设置').format(tfmodel + '.meta'))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", 3,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))
    im_names = ["A_left.jpg","A_right.jpg","B_left.jpg","B_right.jpg"]
    boxes_list = list()
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('开始检测{}'.format(im_name))
        boxes_list.append(demo(sess, net, im_name))
        save_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure",
                                                                        "static/res_pictures/result/") + flag + "/"+im_name
        # boxes_info_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure",
        #                                                                 "static/res_pictures/result/") + flag + "/"+im_name[:-4]+"/"+"boxes_info.xml"
        # doc = minidom.parse(boxes_info_path)

        plt.savefig(save_path,bbox_inches='tight',pad_inches=0)
    print(boxes_list)
    # 用来检测数据是否符合标准,当前因为还没引入靶标，所以暂不启用
    # if not info_check(boxes_list):
    #     return False
    #用来将数据存储到info文件中
    boxes_saveInfo(flag,boxes_list)
    return True



def info_check(boxes_list):
    ##测试图片数
    if len(boxes_list)!=4:
        return False
    ##测试类别数
    numsOf_classes_AL = len(boxes_list[0])
    numsOf_classes_AR = len(boxes_list[1])
    numsOf_classes_BL = len(boxes_list[2])
    numsOf_classes_BR = len(boxes_list[3])
    if numsOf_classes_AL!=2 or numsOf_classes_AR!=2 or numsOf_classes_BL!=2 or numsOf_classes_BR!=2:
        return False
    ##测试靶标框数
    numsOf_targets_AL = len(boxes_list[0][1])
    numsOf_targets_AR = len(boxes_list[1][1])
    numsOf_targets_BL = len(boxes_list[2][1])
    numsOf_targets_BR = len(boxes_list[3][1])
    if numsOf_targets_AL!=numsOf_targets_AR or numsOf_targets_BL!=numsOf_targets_BR:
        return False
    ##测试安装孔数
    numsOf_holes_AL = len(boxes_list[0][0])
    numsOf_holes_AR = len(boxes_list[1][0])
    numsOf_holes_BL = len(boxes_list[2][0])
    numsOf_holes_BR = len(boxes_list[3][0])
    if numsOf_holes_AL!=numsOf_holes_AR or numsOf_holes_BL!=numsOf_holes_BR:
        return False
    return True




def boxes_saveInfo(flag,boxes_list):
    ##此函数用来实现自定义的数据结构向boxes_info存储的函数
    im_names = ["A_left","A_right","B_left","B_right"]
    for i in range(4):
        print("开始存储{:s}的数据".format(im_names[i]))
        save_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure",
                                                                        "static/res_pictures/result/") + flag + "/" + im_names[i] + "/boxes_info.xml"
        img_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure",
                                                                        "static/res_pictures/") + im_names[i] + "/" + flag + im_names[i][1:] + ".jpg"
        img = cv2.imread(img_path)
        dom = minidom.parse(save_path)
        root = dom.documentElement
        boxesNode = root.getElementsByTagName('boxes')[0]
        boxesNode.removeChild(root.getElementsByTagName('hole')[0])
        boxesNode.removeChild(root.getElementsByTagName('target')[0])
        boxesNode.appendChild(dom.createElement('hole'))
        boxesNode.appendChild(dom.createElement('target'))
        itemlist_h = root.getElementsByTagName('hole')
        itemlist_t = root.getElementsByTagName('target')
        item_h = itemlist_h[0]
        item_t = itemlist_t[0]
        numsOf_holes = len(boxes_list[i][0])
        # numsOf_targets = len(boxes_list[i][1]) ## 基准板上不一定三个靶标
        for j in range(numsOf_holes):
            hole_box = dom.createElement('hole_box'+str(j))
            x = dom.createElement('x')
            roi_x = int(boxes_list[i][0][j][0])
            x.appendChild(dom.createTextNode(str(roi_x)))
            hole_box.appendChild(x)
            y = dom.createElement('y')
            roi_y = int(boxes_list[i][0][j][1])
            y.appendChild(dom.createTextNode(str(roi_y)))
            hole_box.appendChild(y)
            w = dom.createElement('w')
            roi_w = int(boxes_list[i][0][j][2])
            w.appendChild(dom.createTextNode(str(roi_w)))
            hole_box.appendChild(w)
            h = dom.createElement('h')
            roi_h = int(boxes_list[i][0][j][3])
            h.appendChild(dom.createTextNode(str(roi_h)))
            hole_box.appendChild(h)
            item_h.appendChild(hole_box)
            hole_box_img = img[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
            save_img_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure",
                                                                        "static/res_pictures/result/") + flag + "/" + im_names[i] + "/boxes_img/hole" + str(j) + ".jpg"
            cv2.imwrite(save_img_path,hole_box_img)
        # for j in range(numsOf_targets):
        #     target_box = dom.createElement('target_box'+str(j))
        #     x = dom.createElement('x')
        #     roi_x = int(boxes_list[i][1][j][0])
        #     x.appendChild(dom.createTextNode(str(roi_x)))
        #     target_box.appendChild(x)
        #     y = dom.createElement('y')
        #     roi_y = int(boxes_list[i][1][j][1])
        #     y.appendChild(dom.createTextNode(str(roi_y)))
        #     target_box.appendChild(y)
        #     w = dom.createElement('w')
        #     roi_w = int(boxes_list[i][1][j][2])
        #     w.appendChild(dom.createTextNode(str(roi_w)))
        #     target_box.appendChild(w)
        #     h = dom.createElement('h')
        #     roi_h = int(boxes_list[i][1][j][3])
        #     h.appendChild(dom.createTextNode(str(roi_h)))
        #     target_box.appendChild(h)
        #     item_t.appendChild(target_box)
        #     target_box_img = img[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
        #     save_img_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure",
        #                                                                 "static/res_pictures/result/") + flag + "/" + im_names[i] + "/boxes_img/target" + str(j) + ".jpg"
        #     cv2.imwrite(save_img_path,target_box_img)
        with open(save_path,'w') as fp:
            dom.writexml(fp)


##图像处理函数
##输入：某一图像路径；图像类别
##输出：保存处理后的图像；保存处理后的形心坐标点数据
def img_process(imgPath,class_of_img="hole"):
    ##核心，数字图像处理算法
    res = [0.0,0.0,0.0,0] ##0:形心横坐标 1:形心纵坐标 2:形心圆半径 3:拟合度评分
    img = cv2.imread(imgPath)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##待定准对象为安装孔时
    if class_of_img=="hole":
        max_NMS = 50 ##高低阈值可能以后需要调试
        TL = max_NMS/2
        TH = max_NMS
        imedge = cv2.Canny(imgray,TL,TH)
        param1 = TH ##跟着上面的高低阈值
        param2 = 35
        maxRadius = 90 ##后期调试
        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1, param2=param2, minRadius=0,
                                   maxRadius=maxRadius)
        while circles is None:
            param2 += -1
            circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1, param2=param2, minRadius=0,
                                       maxRadius=maxRadius)
        while circles.shape[1] > 1:
            param2 += 1
            circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1, param2=param2, minRadius=0,
                                       maxRadius=maxRadius)
        if circles is not None:
            x = int(circles[0,0,0])
            y = int(circles[0,0,1])
            r = int(circles[0,0,2])
            # print(x)
            # print(y)
            # print(r)
            score = 0
            if imedge[y,x-r]!=0:
                score+=1
            if imedge[y,x+r]!=0:
                score+=1
            for i in range(x-r+1,x+r):
                w = abs(x-i)
                h = round((r**2-w**2)**0.5)
                if imedge[y+h,i]!=0:
                    score+=1
                if imedge[y-h,i]!=0:
                    score+=1
            res[0] = circles[0,0,0]
            res[1] = circles[0,0,1]
            res[2] = circles[0,0,2]
            res[3] = score
        savePath = imgPath[:-9]+"p_"+imgPath[-9:]
        saveImg = cv2.line(img,(x,y+20),(x,y-20),(0,255,0),thickness=2)
        saveImg = cv2.line(saveImg,(x+20,y),(x-20,y),(0,255,0),thickness=2)
        saveImg = cv2.circle(saveImg,(x,y),3,(0,0,255),thickness=-1)
        # print("===="+str(x))
        # print("===="+str(y))
        cv2.imwrite(savePath,saveImg)



    ##待定准对象为视觉靶标时
    # elif class_of_img=="target":

    return res
