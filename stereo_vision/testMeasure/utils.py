from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import stereo_vision.testMeasure._init_paths ##这里还要改回来
import _init_paths
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


CLASSES = ('__background__','hole')

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
                                                                                "default/res101_faster_rcnn_iter_55000.ckpt")
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\n请检查output路径设置').format(tfmodel + '.meta'))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", 2,
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


def isqrt(n):
    if n <= 1:
        return n
    lo = 0
    hi = n >> 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        sq = mid * mid
        if sq == n:
            return mid
        elif sq < n:
            lo = mid + 1
            result = mid
        else:
            hi = mid - 1
    return result
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


#############################################################################
####################三维重构部分###############################################
# def base_point(file_dir, img_dir):
#     # todo 靶标 此函数为了获取黑色靶标点
#     img = cv2.imread(img_dir)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.medianBlur(gray, 7)
#     # todo  阈值处理
#     ret, blur = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
#     canny = cv2.Canny(blur, 0, 60, apertureSize=3)
#     ret, contours, hierachy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     max = 0
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > max:
#             max = area
#             p = cnt
#     M = cv2.moments(p)
#     if M['m00'] != 0.0:
#         cx = int(M['m10'] / M['m00'])
#         cy = int(M['m01'] / M['m00'])
#     return (cx,cy)


# todo   直接出三维坐标
def xy2xyz(uvLeft, uvRight):

    root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeRestruction','checkboard_img_dir')
    left_path = os.path.join(root_path,'left_single_calibration.xml')
    right_path = os.path.join(root_path,'right_single_calibration.xml')

    fs_left = cv2.FileStorage(left_path, cv2.FileStorage_READ)
    fs_right = cv2.FileStorage(right_path, cv2.FileStorage_READ)

    rvecs_left = fs_left.getNode('rvecs_1').mat()
    tvecs_left = fs_left.getNode('tvecs_1').mat()
    rvecs_right = fs_right.getNode('rvecs_1').mat()
    tvecs_right = fs_right.getNode('tvecs_1').mat()

    # 读取左相机旋转平移矩阵
    # Rodrigues 会输出两个矩阵，一个3*3, 一个3*9
    # np.concatenate 数组拼接
    # 一般axis = 0，就是对该轴向的数组进行操作，操作方向是另外一个轴，即axis=1。
    # 传入的数组必须具有相同的形状，这里的相同的形状可以满足在拼接方向axis轴上数组间的形状一致即可
    # numpy dot获取两个元素a,b的乘积
    mLeftRotation =np.zeros((3, 3), np.float32)
    mLeftTranslation = np.zeros((3, 1), np.float32)
    mLeftRT = np.zeros((3, 4), np.float32)     # 左相机M矩阵

    temp = cv2.Rodrigues(rvecs_left, mLeftRotation)
    temp1 = cv2.Rodrigues(tvecs_left, mLeftTranslation)
    mLeftRotation = temp[0]
    mLeftTranslation = temp1[0]

    mLeftRT = np.concatenate((mLeftRotation, mLeftTranslation), axis=1)
    mLeftIntrinsic = np.zeros((3, 3), np.float32)
    mLeftIntrinsic = fs_left.getNode('mtx').mat()

    mLeftM = mLeftIntrinsic.dot(mLeftRT)

    # 读取右相机旋转平移矩阵
    mRightRotation = np.zeros((3, 3), np.float32)
    mRightTranslation = np.zeros((3, 1), np.float32)
    temp = cv2.Rodrigues(rvecs_right, mRightRotation)
    temp1 = cv2.Rodrigues(tvecs_right, mRightTranslation)

    mRightRotation = temp[0]
    mRightTranslation = temp1[0]

    mRightRT = np.zeros((3, 4), np.float32)
    mRightRT = np.concatenate((mRightRotation, mRightTranslation), axis=1)
    mRightIntrinsic = np.zeros((3, 3), np.float32)
    mRightIntrinsic = fs_right.getNode('mtx').mat()

    mRightM = mRightIntrinsic.dot(mRightRT)


    A = np.zeros((4, 3), np.float32)

    mLeftM[2, 0] = 1

    A[0, 0] = uvLeft[0] * mLeftM[2, 0] - mLeftM[0, 0]
    A[0, 1] = uvLeft[0] * mLeftM[2, 1] - mLeftM[0, 1]
    A[0, 2] = uvLeft[0] * mLeftM[2, 2] - mLeftM[0, 2]

    A[1, 0] = uvLeft[1] * mLeftM[2, 0] - mLeftM[1, 0]
    A[1, 1] = uvLeft[1] * mLeftM[2, 1] - mLeftM[1, 1]
    A[1, 2] = uvLeft[1] * mLeftM[2, 2] - mLeftM[1, 2]

    A[2, 0] = uvRight[0] * mRightM[2, 0] - mRightM[0, 0]
    A[2, 1] = uvRight[0] * mRightM[2, 1] - mRightM[0, 1]
    A[2, 2] = uvRight[0] * mRightM[2, 2] - mRightM[0, 2]

    A[3, 0] = uvRight[1] * mRightM[2, 0] - mRightM[1, 0]
    A[3, 1] = uvRight[1] * mRightM[2, 1] - mRightM[1, 1]
    A[3, 2] = uvRight[1] * mRightM[2, 2] - mRightM[1, 2]

    B = np.zeros((4, 1), np.float32)

    B[0, 0] = mLeftM[0, 3] - uvLeft[0] * mLeftM[2, 3]
    # B[0, 1]
    B[1, 0] = mLeftM[1, 3] - uvLeft[1] * mLeftM[2, 3]
    B[2, 0] = mRightM[0, 3] - uvRight[0] * mRightM[2, 3]
    B[3, 0] = mRightM[1, 3] - uvRight[1] * mRightM[2, 3]

    XYZ = np.zeros((3, 1),np.float32)
    # cv2.solve用于解决 A*X = B
    # XYZ相当于输出的解决方案

    # cv2.SVD_FULL_UV ?
    # cv2.SVD_NO_UV
    # cv2.SVD_Modify_A
    cv2.solve(A, B, XYZ, cv2.SVD_FULL_UV)

    #cv2.solveP3P()
    #cv2.solvePnP()

    return XYZ


def  point_undistort(point,flag,end='A'):
    if end=='A':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/checkboard_img_dir/A_')
    elif end=='B':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/checkboard_img_dir/B_')
    left_path = root_path+'left_single_calibration.xml'
    right_path = root_path+'right_single_calibration.xml'
    # print(left_path)
    # print(right_path)
    if flag == 'left':
        fs_left = cv2.FileStorage(left_path, cv2.FileStorage_READ)
        Intrinsic = fs_left.getNode('mtx').mat()
        Dist = fs_left.getNode('dist').mat()

    elif flag == 'right':
        fs_right = cv2.FileStorage(right_path, cv2.FileStorage_READ)
        Intrinsic = fs_right.getNode('mtx').mat()
        Dist = fs_right.getNode('dist').mat()
    # print(Intrinsic)
    # print(Dist)
    src = np.array([[point[0],point[1]]], dtype=np.float)
    src = np.array(np.reshape(src,(len(src),1,2)),dtype=float)
    dst = cv2.undistortPoints(src,Intrinsic,Dist)

    fx = Intrinsic[0][0]
    fy = Intrinsic[1][1]
    u0 = Intrinsic[0][2]
    v0 = Intrinsic[1][2]

    # 消除畸变后重新将相机坐标系转换到图像像素坐标系
    x__ = dst[0][0][0] * fx + u0
    y__ = dst[0][0][1] * fy + v0

    return [x__, y__]


def  pixel2cam(point,intrinsic):

    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    u0 = intrinsic[0][2]
    v0 = intrinsic[1][2]

    # 图像像素坐标转换到相机坐标系
    u  = point[0]
    v  = point[1]

    #x  = (u- u0)/fy
    #y  = (v- v0)/fx

    x = (u - u0) / fx
    y = (v - v0) / fy

    xy = np.array([[x],[y]],dtype=np.float)

    return xy

def xy2xyz1(uvLeft, uvRight,end='A'):
    if end=='A':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure','static/checkboard_img_dir/A_')
    elif end=='B':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure','static/checkboard_img_dir/B_')
    left_path = root_path+'left_single_calibration.xml'
    right_path = root_path+'right_single_calibration.xml'
    stereo_path = root_path+'stereo_calibration.xml'
    # print(left_path)
    # print(right_path)
    # print(stereo_path)

    fs_left = cv2.FileStorage(left_path, cv2.FileStorage_READ)
    fs_right = cv2.FileStorage(right_path, cv2.FileStorage_READ)
    fs_stereo = cv2.FileStorage(stereo_path, cv2.FileStorage_READ)

    mLeftIntrinsic = fs_left.getNode('mtx').mat()
    mRightIntrinsic = fs_right.getNode('mtx').mat()
    R_matrix  = fs_stereo.getNode('R_matrix').mat()
    T_matrix  = fs_stereo.getNode('T_matrix').mat()


    XL, YL = uvLeft[0],  uvLeft[1]
    XR, YR = uvRight[0], uvRight[1]


    camera_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    projector_matrix = np.hstack((R_matrix, T_matrix))

    L = np.array([[XL], [YL]],dtype=np.float)
    R = np.array([[XR], [YR]], dtype=np.float)

    L = pixel2cam(L, mLeftIntrinsic)
    R = pixel2cam(R, mRightIntrinsic)

    points = cv2.triangulatePoints(camera_matrix, projector_matrix, L, R)
    points = [points[0] / points[3], points[1] / points[3], points[2] / points[3]]


    return points


# todo   修改  此处 需要进行 三个数据的传入
def  three_dimension_restruction(uvLeft, uvRight, epoch_name):

    left_pic_path  = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/left/2019-06-06-16:38:59/2019-06-06-16:38:59_left.jpg'
    right_pic_path = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/right/2019-06-06-16:38:59/2019-06-06-16:38:59_right.jpg'

    root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeRestruction', 'checkboard_img_dir')   # 找到双目标定的数据
    stereo_path = os.path.join(root_path,'stereo_calibration.xml')

    stereo_calibration_fs = cv2.FileStorage(stereo_path,cv2.FileStorage_READ)

    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    blockSize = 11

    # 创建一个新的stereoSGBM
    # mindisparity最小可能的差异值
    # numDisparities 最大最小差异差值
    # P1=8 * 3 * window_size ** 2= 24*11*11
    stereo = cv2.StereoSGBM_create(minDisparity=2,
                                   numDisparities=128,
                                   blockSize=11,
                                   uniquenessRatio=5,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    # 创建WLS滤波器
    stereoR = cv2.ximgproc.createRightMatcher(stereo)

    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # 读取左右相机图片和左右重映射矩阵
    left_pic = cv2.imread(left_pic_path)
    right_pic = cv2.imread(right_pic_path)
    # todo  获取左右Stereo_map

    Left_Stereo_Map_0 = stereo_calibration_fs.getNode('left_stereo_map0').mat()
    Left_Stereo_Map_1 = stereo_calibration_fs.getNode('left_stereo_map1').mat()
    Right_Stereo_Map_0 = stereo_calibration_fs.getNode('right_stereo_map0').mat()
    Right_Stereo_Map_1 = stereo_calibration_fs.getNode('right_stereo_map1').mat()




    print(Left_Stereo_Map_0)
    print(Left_Stereo_Map_1)
    print(Right_Stereo_Map_0)
    print(Right_Stereo_Map_1)
    # 修正左右相机图片畸变

    left_remap = cv2.remap(left_pic, Left_Stereo_Map_0, Left_Stereo_Map_1, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)



    cv2.imwrite('/home/cx/Desktop/remote_image/code/undistort/left_remap'+epoch_name+'.jpg',left_remap)

    right_remap = cv2.remap(right_pic, Right_Stereo_Map_0, Right_Stereo_Map_1, cv2.INTER_LANCZOS4,
                            cv2.BORDER_CONSTANT, 0)

    cv2.namedWindow('', 0)
    cv2.resizeWindow('', (800, 600))
    cv2.imshow('', np.hstack([left_pic, right_pic]))
    #cv2.imshow('', filteredImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('/home/cx/Desktop/remote_image/code/undistort/right_remap'+epoch_name+'.jpg',right_remap)

    # 转变为灰度图
    """
    right_gray = cv2.cvtColor(right_remap, cv2.COLOR_BGR2GRAY)
    left_gray = cv2.cvtColor(left_remap, cv2.COLOR_BGR2GRAY)

    # 计算视差  找到 disperity
    dispL1 = stereo.compute(left_gray, right_gray)
    dispR = stereo.compute(right_gray, left_gray)
    # 找到depth
    Q = stereo_calibration_fs.getNode('rective_stereo').mat()


    dispL = np.int16(dispL1)
    dispR = np.int16(dispR)


    # 滤波， 归一化  最小二乘滤波器
    filteredImg = wls_filter.filter(dispL, left_gray, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    #cv2.namedWindow('', 0)
    #cv2.resizeWindow('', (800, 600))
    #cv2.imshow('', np.hstack([dispL, dispR]))
    #cv2.imshow('',filteredImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




    disp = ((dispL.astype(np.float32) / 16) - min_disp) / num_disp
    dispR = cv2.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    # 给定核的大小
    # 腐蚀处理
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

    dispc = (closing - closing.min()) * 255
    dispC = dispc.astype(np.uint8)
    # todo 查看这几个有啥差别
    disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)

    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
    jet_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)
    bone_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_BONE)
    img = cv2.cvtColor(disp_Color, cv2.COLOR_BGR2GRAY)


    white_pixel_count = 0  # Variable to store the total strength of all pixels
    white_pixels = 0.0
    offset = 21951047  # Summation of pixel strength for empty box


    # todo 修改pixel 的位置
    #px = img[uvLeft[1], uvLeft[0]]  # access the particular pixel
    #white_pixels = white_pixels + px

    #int(3648), int(5472)
    for y in range(1, 3648):
        for x in range(1, 5472):
            px = img[y, x]  # access the particular pixel
            white_pixels = white_pixels + px
    px = img[uvLeft[1], uvLeft[0]]
    print('px',px)
    print('Current Pixel Strength =', white_pixels - offset)
    covered_space = ((white_pixels - 21951047) / 56000000.0)
    # 56000000.0 is the total pixel value for completely filled cargo box
    print('Covered Space in % =', str(covered_space * 100)[:5])
    # todo  读取Q矩阵


    # 三维坐标数据矩阵是三通道浮点型的
    # 得到一副映射图，图像大小与视差图相同，且每个像素具有三个通道，分别存储了该像素位置在相机坐标系下的三维点坐标在x, y, z,三个轴上的值，
    # 即每个像素的在相机坐标系下的三维坐标。
    # point_left的作用？
    # todo  dispL 需要换成某点的视差

    three_coordinates = cv2.reprojectImageTo3D(dispL1, Q)
    three_coordinates = three_coordinates.reshape(-1, 3)
    colors = cv2.cvtColor(left_remap,cv2.COLOR_BGR2RGB)
    colors = colors.reshape(-1, 3)
    mask = three_coordinates[:, 2] > three_coordinates[:, 2].min()
    coords = three_coordinates[mask]
    colors = colors[mask]
    ply_header = (
        '''ply
        format ascii 1.0
        element vertex {vertex_count}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        ''')

    points = np.hstack([three_coordinates, colors])
    output_file = '/home/cx/Desktop/remote_image/code/111.txt'
    with open(output_file, 'w') as outfile:
        outfile.write(ply_header.format(
            vertex_count=len(three_coordinates)))
        np.savetxt(outfile, points, '%f %f %f %d %d %d')

    print(colors)

    cv2.namedWindow('', 0)
    cv2.resizeWindow('', (800, 600))
    #cv2.imshow('', np.hstack([dispL, dispR]))
    cv2.imshow('',coords)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """

    return   1


# def stereo_Calibration_for_use(left_name, right_name):
#
#     calibration_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
#                                                                            'checkboard_img_dir/' + 'stereo_calibration.xml')
#     stereo_calibration_fs = cv2.FileStorage(calibration_path,cv2.FileStorage_READ)
#
#     base_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/zj_pictures/stereo_calibration/')
#
#     # list(range(1000))
#     Left_Stereo_Map = ['', '']
#     Right_Stereo_Map = ['', '']
#     Left_Stereo_Map[0] = stereo_calibration_fs.getNode('left_stereo_map0').mat()
#     Left_Stereo_Map[1] = stereo_calibration_fs.getNode('left_stereo_map1').mat()
#     Right_Stereo_Map[0] = stereo_calibration_fs.getNode('right_stereo_map0').mat()
#     Right_Stereo_Map[1] = stereo_calibration_fs.getNode('right_stereo_map1').mat()
#     Q                   = stereo_calibration_fs.getNode('rective_stereo').mat()
#     stereo_calibration_fs.release()
#
#     img_right = cv2.imread(right_name)
#     img_left = cv2.imread(left_name)
#     img = cv2.remap(img_left, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
#     img2 = cv2.remap(img_right, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)
#
#     # 灰度图准备
#     imgL = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgR = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     # 根据Block Maching方法生成差异图
#     stereo = cv2.StereoBM_create(numDisparities=144, blockSize=5)
#     disparity = stereo.compute(imgL, imgR)
#     disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
#     threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 32., Q)
#
#     min_disp= 2
#
#     num_disp = 130 - min_disp
#     disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
#
#     cv2.imwrite(os.path.join(base_path, 'stereo_remap_left.jpg'), img)
#     cv2.imwrite(os.path.join(base_path, "stereo_remap_right.jpg"), img2)
#     cv2.imwrite(os.path.join(base_path, "disp.jpg"), disparity)
#     cv2.imwrite(os.path.join(base_path, "disparity.jpg"), threeD)
#     cv2.imwrite(os.path.join(base_path, "depth.jpg"), disp)



# todo  匹配   修改bmp为jpg
# def   jixian_function(epoch_name):
#
#         root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeRestruction', 'static/zj_pictures/')
#         #left_pic_path = os.path.join(root_path,'left',epoch_name,'left.jpg')
#         #right_pic_path = os.path.join(root_path, 'right', epoch_name, 'right.jpg')
#
#         left_pic_path = '/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/2019-09-01-17:50:00_left.jpg'
#         right_pic_path = '/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/2019-09-01-17:50:00_right.jpg'
#
#         left = cv2.imread(left_pic_path,0)
#         right = cv2.imread(right_pic_path,0)
#
#         sift = cv2.xfeatures2d.SIFT_create()
#
#         kp1,des1 = sift.detectAndCompute(left,None)
#         kp2,des2 = sift.detectAndCompute(right,None)
#
#         cbrow = 9  # the number of the innner corner must be  correct
#         cbcol = 13
#
#         objp = np.zeros((cbrow * cbcol, 3), np.float32)
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#         objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)
#         objpoints = []  # 用来存放三维点
#         imgpoints = []  # 用来存放图像平面中的二维点
#         pts1 = []
#         pts2 = []
#
#
#         corner_image_name = 1
#
#         corner_img_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'corner_img_dir/')
#
#         # corner_img_dir = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/corner_img_dir/'
#
#         ret, corners1 = cv2.findChessboardCorners(left, (9, 13), None)
#         """角点精确化迭代过程的终止条件"""
#         """执行亚像素级角点检测"""
#
#         corners1 = cv2.cornerSubPix(left, corners1, (11, 11), (-1, -1), criteria)
#         objpoints.append(objp)
#         pts1.append(corners1)
#
#         ret, corners2 = cv2.findChessboardCorners(right, (9, 13), None)
#         """角点精确化迭代过程的终止条件"""
#         """执行亚像素级角点检测"""
#
#         corners2 = cv2.cornerSubPix(right, corners2, (11, 11), (-1, -1), criteria)
#         objpoints.append(objp)
#         pts2.append(corners2)
#
#
#         FLANN_INDEX_KDTREE = 0
#
#         index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks= 50)
#
#         flann = cv2.FlannBasedMatcher(index_params,search_params)
#
#         print(des1)
#         matches = flann.knnMatch(des1,des2,k=2)
#
#         good = []
#
#         for i ,(m,n) in enumerate(matches):
#              if  m.distance < 0.8*n.distance:
#                  good.append(m)
#                 # print('kp2[m.trainIdx].pt', kp2[m.trainIdx].pt)
#                  pts2.append(kp2[m.trainIdx].pt)
#                 # print('kp1[m.queryIdx].pt', kp1[m.queryIdx].pt)
#                  pts1.append(kp1[m.queryIdx].pt)
#         pts1 = np.int32(pts1)
#         pts2 = np.int32(pts2)
#
#         F = find_Fmatrix(epoch_name)
#
#          # pts1  <class 'numpy.ndarray'>   (3648, 5472, 3)
#
#          #pts1 = np.zeros((1,1,3))
#          #pts1[0][0] = (981,1397)
#          #print(pts1,type)
#          #print(pts1,type(pts1))
#
#         lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
#         lines1 = lines1.reshape(-1,3)
#
#         img5,img6 = drawlines(left,right,lines1,pts1,pts2)
#
#
#         lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),2,F)
#         lines2 = lines2.reshape(-1,3)
#
#         img3,img4 = drawlines(right,left,lines2,pts2,pts1)
#
#
#         cv2.imwrite('/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/img3.jpg', img3)
#         cv2.imwrite('/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/img4.jpg', img4)
#         cv2.imwrite('/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/img5.jpg', img5)
#         cv2.imwrite('/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/img6.jpg', img6)
#
#
#         cv2.namedWindow('',0)
#         cv2.resizeWindow('',(800,600))
#         cv2.imshow('',img5)
#         cv2.imshow('', img3)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         plt.subplot(121),plt.imshow(img5)
#         plt.subplot(122),plt.imshow(img3)
#         plt.show()



def   find_Fmatrix(epoch_name): #涉及函数1

    calibration_path_A = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure',
                                                                        'static/checkboard_img_dir/' + 'A_stereo_calibration.xml')
    calibration_path_B = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure',
                                                                        'static/checkboard_img_dir/' + 'B_stereo_calibration.xml')
    stereo_calibration_fs_A = cv2.FileStorage(calibration_path_A, cv2.FileStorage_READ)
    stereo_calibration_fs_B = cv2.FileStorage(calibration_path_B, cv2.FileStorage_READ)

    F_A = stereo_calibration_fs_A.getNode('fundamental_matrix').mat()
    F_B = stereo_calibration_fs_B.getNode('fundamental_matrix').mat()
    stereo_calibration_fs_A.release()
    stereo_calibration_fs_B.release()

    F_A = F_A.T
    F_B = F_B.T
    # print(F_A)
    # print(F_B)
    return  F_A,F_B


# def find_Fmatrix1(epoch_name):
#
#     calibration_path = os.path.dirname(os.path.realpath(__file__)).replace('threeRestruction',
#                                                                            'checkboard_img_dir/' + 'stereo_calibration.xml')
#     stereo_calibration_fs = cv2.FileStorage(calibration_path, cv2.FileStorage_READ)
#     R = stereo_calibration_fs.getNode('R_matrix').mat()
#     T = stereo_calibration_fs.getNode('T_matrix').mat()
#     stereo_calibration_fs.release()
#     left_single_path =calibration_path.replace('stereo_calibration','left_single_calibration')
#     left_calibration_fs = cv2.FileStorage(left_single_path, cv2.FileStorage_READ)
#     Al = left_calibration_fs.getNode('mtx').mat()
#     left_calibration_fs.release()
#     right_single_path =calibration_path.replace('stereo_calibration','right_single_calibration')
#     right_calibration_fs = cv2.FileStorage(right_single_path, cv2.FileStorage_READ)
#     Ar = right_calibration_fs.getNode('mtx').mat()
#     right_calibration_fs.release()
#
#     TZ = T[2]
#     TX = T[0]
#     TY = T[1]
#
#     S = np.zeros((3,3),np.float32)
#     S[0][0] = 0
#     S[0][1] = -TZ
#     S[0][2] = TY
#     S[1][0] = TZ
#     S[1][1] = 0
#     S[1][2] = -TX
#     S[2][0] = -TY
#     S[2][1] = TX
#     S[2][2] = 0
#     Ar = np.linalg.inv(Ar.transpose())
#     Al = np.linalg.inv(Al)
#     Ar_S= np.dot(Ar,S)
#     R_Al = np.dot(R,Al)
#     F = np.dot(Ar_S,R_Al)
#
#     return F


def  point2jixian(pts3,F):  ##对极约束
    pts3 = np.int32(pts3)
    line = cv2.computeCorrespondEpilines(pts3.reshape(-1,1,2),2,F)
    line= line.reshape(-1,3)
    return  line


def drawlines(img1,img2,lines,pts1,pts2):

    r,c = img1.shape[0],img1.shape[1]
    print('r',r)
    print('c',c)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    for  r ,pt1,pt2  in  zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int,[0,-r[2]/r[1]])
        x1,y1 = map(int,[c,-(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1,(x0,y0),(x1,y1),color,1)
        img1 = cv2.circle(img1,tuple(pt1),15,color,-1)
        img2 = cv2.circle(img2, tuple(pt2), 15, color, -1)

    return img1 ,img2

# todo   一个点对应一条直线，对应左面图像上的点找到在右面图像上的直线
# (x-x1)/(x2-x1)=(y-y1)/(y2-y1)
# 0 = (y2-y1)/(x2-x1)*x -y -(y2-y1)*x1/(x2-x1)  0 = Ax+By+C
def find_line(lines,pts1,flag=1):

    # 在img1上面找极线
    # lines是由于pts2求得的
    # pts1是img1中的点
    # pts1 = [(702.858154296875, 1142.3895263671875)]
    # a, c = img1.shape[0], img1.shape[1]
    pts1 = np.int32(pts1)
    x0, y0 = 0,0
    x1, y1 = 0,0

    for  r ,pt1  in  zip(lines,pts1):

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1,y1 = map(int,[pts1[0],-(r[2]+r[0]*pts1[0])/r[1]])

        flag += 1
        A = r[0]
        B = r[1]
        C = r[2]

    #right_pic_path = '/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/2019-09-01-17:50:00_right.jpg'
    #color = tuple(np.random.randint(0, 255, 3).tolist())
    #img1 = cv2.imread(right_pic_path)
    #img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 5)
    #img1 = cv2.circle(img1, (pts1[0],pts1[1]), 15, color, -1)
    #cv2.imwrite('/home/cx/Desktop/部分代码修改/postgraduate-master/result/exp/2019-09-01-17:50:00_check'+str(flage)+str(A)+'_'+str(B)+'_'+str(C)+'.jpg',img1)

    return  A,B,C


# pts2 = [(702.858154296875, 1142.3895263671875)]
def  compute_point_distance(pts,A,B,C):  ##计算点到直线的距离

    x0 =  pts[0]
    y0 =  pts[1]
    #print(x0,y0,A,B,C)
    d = math.fabs((A*x0+B*y0+C)/(math.sqrt(A*A+B*B)))
    # print('点到极限的距离：',d)
    return  d


# 函数的作用:left_point在right_points里面的最匹配点
def  min_distance_pnt(right_points,left_point,F):

    distance_init = math.inf
    obj_point = None
    index = 0
    for  i , item in enumerate(right_points):
        pts3 = [0,0]
        pts3[0] = left_point[0]
        pts3[1] = left_point[1]
        line = point2jixian(pts3, F)
        pts = [0,0]
        pts[0] = item[0]
        pts[1] = item[1]
        A,B,C  = find_line(line,pts,i)

        distance = compute_point_distance(pts, A, B, C)

        if distance < distance_init:

           distance_init = distance
           obj_point = item
           index = i
    return  index,obj_point

## 此函数的输出格式也需要修改
def  creat_point_pairs(right_points,left_points,F):
     all_pairs = []
     for left_point in  left_points:
        index,right_point =  min_distance_pnt(right_points, left_point,F) ##找出对应当前左点的最佳右点
        # todo 查看right_point数据格式
        print(len(right_points))
        right_points.pop(index)
        # pair['left_point'] = left_point
        # pair['right_point'] = right_point
        # print('left_point', left_point)
        # print('right_point',right_point)
        left_x = left_point[0]
        left_y = left_point[1]
        right_x = right_point[0]
        right_y = right_point[1]
        radius = (left_point[2]+right_point[2])//2
        score = (left_point[3]+right_point[3])//2
        all_pairs.append([[left_x,left_y],[right_x,right_y],radius,score])
     print("=====")
     return all_pairs

# 修改后把匹配好的左右点存入points_info.xml
def save_pairs_file(epoch_name,pairs,end='A',hot="hole"):
    ##主要步骤是先擦除原始数据，再往里面写数据
    root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/')
    file_path = os.path.join(root_path,epoch_name+"/points_info.xml")
    dom = minidom.parse(file_path)
    root = dom.documentElement
    if end=='A':
        if hot=="hole":
            theNode = root.getElementsByTagName("hole")[0]
            theNode.removeChild(root.getElementsByTagName("pairs")[0])
            theNode.removeChild(root.getElementsByTagName("threeD")[0])
            p = dom.createElement("pairs")
            theNode.appendChild(p)
            theNode.appendChild(dom.createElement("threeD"))
            i=0
            for pair in pairs:
                thePair = dom.createElement("pair" + str(i))
                left_x = dom.createElement("left_x")
                left_x.appendChild(dom.createTextNode(str(pair[0][0])))
                thePair.appendChild(left_x)
                left_y = dom.createElement("left_y")
                left_y.appendChild(dom.createTextNode(str(pair[0][1])))
                thePair.appendChild(left_y)
                right_x = dom.createElement("right_x")
                right_x.appendChild(dom.createTextNode(str(pair[1][0])))
                thePair.appendChild(right_x)
                right_y = dom.createElement("right_y")
                right_y.appendChild(dom.createTextNode(str(pair[1][1])))
                thePair.appendChild(right_y)
                radius = dom.createElement("radius")
                radius.appendChild(dom.createTextNode(str(pair[2])))
                thePair.appendChild(radius)
                score = dom.createElement("score")
                score.appendChild(dom.createTextNode(str(pair[3])))
                thePair.appendChild(score)
                p.appendChild(thePair)
                i+=1
        elif hot=="target":
            theNode = root.getElementsByTagName("target")[0]
            theNode.removeChild(root.getElementsByTagName("pairs")[1])
            theNode.removeChild(root.getElementsByTagName("threeD")[1])
            p = dom.createElement("pairs")
            theNode.appendChild(p)
            theNode.appendChild(dom.createElement("threeD"))
            i = 0
            for pair in pairs:
                thePair = dom.createElement("pair" + str(i))
                left_x = dom.createElement("left_x")
                left_x.appendChild(dom.createTextNode(str(pair[0][0])))
                thePair.appendChild(left_x)
                left_y = dom.createElement("left_y")
                left_y.appendChild(dom.createTextNode(str(pair[0][1])))
                thePair.appendChild(left_y)
                right_x = dom.createElement("right_x")
                right_x.appendChild(dom.createTextNode(str(pair[1][0])))
                thePair.appendChild(right_x)
                right_y = dom.createElement("right_y")
                right_y.appendChild(dom.createTextNode(str(pair[1][1])))
                thePair.appendChild(right_y)
                radius = dom.createElement("radius")
                radius.appendChild(dom.createTextNode(str(pair[2])))
                thePair.appendChild(radius)
                score = dom.createElement("score")
                score.appendChild(dom.createTextNode(str(pair[3])))
                thePair.appendChild(score)
                p.appendChild(thePair)
                i += 1
    elif end=='B':
        if hot=="hole":
            theNode = root.getElementsByTagName("hole")[1]
            theNode.removeChild(root.getElementsByTagName("pairs")[2])
            theNode.removeChild(root.getElementsByTagName("threeD")[2])
            p = dom.createElement("pairs")
            theNode.appendChild(p)
            theNode.appendChild(dom.createElement("threeD"))
            i = 0
            for pair in pairs:
                thePair = dom.createElement("pair" + str(i))
                left_x = dom.createElement("left_x")
                left_x.appendChild(dom.createTextNode(str(pair[0][0])))
                thePair.appendChild(left_x)
                left_y = dom.createElement("left_y")
                left_y.appendChild(dom.createTextNode(str(pair[0][1])))
                thePair.appendChild(left_y)
                right_x = dom.createElement("right_x")
                right_x.appendChild(dom.createTextNode(str(pair[1][0])))
                thePair.appendChild(right_x)
                right_y = dom.createElement("right_y")
                right_y.appendChild(dom.createTextNode(str(pair[1][1])))
                thePair.appendChild(right_y)
                radius = dom.createElement("radius")
                radius.appendChild(dom.createTextNode(str(pair[2])))
                thePair.appendChild(radius)
                score = dom.createElement("score")
                score.appendChild(dom.createTextNode(str(pair[3])))
                thePair.appendChild(score)
                p.appendChild(thePair)
                i += 1
        elif hot=="target":
            theNode = root.getElementsByTagName("target")[1]
            theNode.removeChild(root.getElementsByTagName("pairs")[3])
            theNode.removeChild(root.getElementsByTagName("threeD")[3])
            p = dom.createElement("pairs")
            theNode.appendChild(p)
            theNode.appendChild(dom.createElement("threeD"))
            i = 0
            for pair in pairs:
                thePair = dom.createElement("pair" + str(i))
                left_x = dom.createElement("left_x")
                left_x.appendChild(dom.createTextNode(str(pair[0][0])))
                thePair.appendChild(left_x)
                left_y = dom.createElement("left_y")
                left_y.appendChild(dom.createTextNode(str(pair[0][1])))
                thePair.appendChild(left_y)
                right_x = dom.createElement("right_x")
                right_x.appendChild(dom.createTextNode(str(pair[1][0])))
                thePair.appendChild(right_x)
                right_y = dom.createElement("right_y")
                right_y.appendChild(dom.createTextNode(str(pair[1][1])))
                thePair.appendChild(right_y)
                radius = dom.createElement("radius")
                radius.appendChild(dom.createTextNode(str(pair[2])))
                thePair.appendChild(radius)
                score = dom.createElement("score")
                score.appendChild(dom.createTextNode(str(pair[3])))
                thePair.appendChild(score)
                p.appendChild(thePair)
                i += 1
    with open(file_path, 'w') as fp:
        dom.writexml(fp)

##函数作用：读取文件中所有的
def  read_feature_file1(epoch_name): # 涉及函数2
    ## 想办法把json读取改成xml数据读取
    ## 分别返回AL、AR、BL、BR点的二维坐标（未匹配）列表中的数据格式为[横坐标，纵坐标，半径估算，检测评分]
    root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/')
    file_dir_path = os.path.join(root_path,epoch_name)
    AL_file_path = file_dir_path + "/A_left/boxes_info.xml"
    AR_file_path = file_dir_path + "/A_right/boxes_info.xml"
    BL_file_path = file_dir_path + "/B_left/boxes_info.xml"
    BR_file_path = file_dir_path + "/B_right/boxes_info.xml"
    AL_holes = list()
    AR_holes = list()
    BL_holes = list()
    BR_holes = list()
    AL_targets = list()
    AR_targets = list()
    BL_targets = list()
    BR_targets = list()
    AL_root = minidom.parse(AL_file_path)
    AR_root = minidom.parse(AR_file_path)
    BL_root = minidom.parse(BL_file_path)
    BR_root = minidom.parse(BR_file_path)
    AL_numsOfholes = len(AL_root.documentElement.getElementsByTagName("hole")[0].childNodes)
    AL_numsOftargets = len(AL_root.documentElement.getElementsByTagName("target")[0].childNodes)
    AR_numsOfholes = len(AR_root.documentElement.getElementsByTagName("hole")[0].childNodes)
    AR_numsOftargets = len(AR_root.documentElement.getElementsByTagName("target")[0].childNodes)
    BL_numsOfholes = len(BL_root.documentElement.getElementsByTagName("hole")[0].childNodes)
    BL_numsOftargets = len(BL_root.documentElement.getElementsByTagName("target")[0].childNodes)
    BR_numsOfholes = len(BR_root.documentElement.getElementsByTagName("hole")[0].childNodes)
    BR_numsOftargets = len(BR_root.documentElement.getElementsByTagName("target")[0].childNodes)
    ##开始遍历所有的孔和靶标
    for i in range(AL_numsOfholes):
        box_X = float(AL_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(AL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(AL_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(AL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(AL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(AL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[3].childNodes[0].data)
        AL_holes.append([box_X+img_X,box_Y+img_Y,Radius,Score])
    for i in range(AL_numsOftargets):
        box_X = float(AL_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(AL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(AL_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(AL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(AL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(AL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[3].childNodes[0].data)
        AL_targets.append([box_X+img_X,box_Y+img_Y,Radius,Score])

    for i in range(AR_numsOfholes):
        box_X = float(AR_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(AR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(AR_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(AR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(AR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(AR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[3].childNodes[0].data)
        AR_holes.append([box_X+img_X,box_Y+img_Y,Radius,Score])
    for i in range(AR_numsOftargets):
        box_X = float(AR_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(AR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(AR_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(AR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(AR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(AR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[3].childNodes[0].data)
        AR_targets.append([box_X+img_X,box_Y+img_Y,Radius,Score])

    for i in range(BL_numsOfholes):
        box_X = float(BL_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(BL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(BL_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(BL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(BL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(BL_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[3].childNodes[0].data)
        BL_holes.append([box_X+img_X,box_Y+img_Y,Radius,Score])
    for i in range(BL_numsOftargets):
        box_X = float(BL_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(BL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(BL_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(BL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(BL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(BL_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[3].childNodes[0].data)
        BL_targets.append([box_X+img_X,box_Y+img_Y,Radius,Score])

    for i in range(BR_numsOfholes):
        box_X = float(BR_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(BR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(BR_root.documentElement.getElementsByTagName("hole")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(BR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(BR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(BR_root.documentElement.getElementsByTagName("hole")[1].childNodes[i].childNodes[3].childNodes[0].data)
        BR_holes.append([box_X+img_X,box_Y+img_Y,Radius,Score])
    for i in range(BR_numsOftargets):
        box_X = float(BR_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[0].childNodes[0].data)
        img_X = float(BR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[0].childNodes[0].data)
        box_Y = float(BR_root.documentElement.getElementsByTagName("target")[0].childNodes[i].childNodes[1].childNodes[0].data)
        img_Y = float(BR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[1].childNodes[0].data)
        Radius = float(BR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[2].childNodes[0].data)
        Score = float(BR_root.documentElement.getElementsByTagName("target")[1].childNodes[i].childNodes[3].childNodes[0].data)
        BR_targets.append([box_X+img_X,box_Y+img_Y,Radius,Score])


    return AL_holes,AR_holes,BL_holes,BR_holes,AL_targets,AR_targets,BL_targets,BR_targets



##此函数主要作用是获取文件中的点
def   get_epoch_pairs_points(epoch_name): ##主要函数1


    F_A,F_B = find_Fmatrix(epoch_name) ##涉及函数1

    """
    left_hole_centers, right_hole_centers, left_zj_centers, right_zj_centers = read_feature_file(epoch_name)

    print('left_hole_centers')
    print(left_hole_centers)
    print('right_hole_centers')
    print(right_hole_centers)
    print('left_zj_centers')
    print(left_zj_centers)
    print('right_zj_centers')
    print(right_zj_centers)
    """

    AL_holes, AR_holes, BL_holes, BR_holes, AL_targets, AR_targets, BL_targets, BR_targets = read_feature_file1(epoch_name) ##涉及函数2
    A_holes_pairs = creat_point_pairs(AR_holes, AL_holes, F_A) ##涉及函数3
    B_holes_pairs = creat_point_pairs(BR_holes, BL_holes, F_B)
    A_targets_pairs = creat_point_pairs(AR_targets, AL_targets, F_A)
    B_targets_pairs = creat_point_pairs(BR_targets, BL_targets, F_B)
    print('A_holes_all_pairs：',A_holes_pairs)
    print('B_holes_all_pairs：', B_holes_pairs)
    print('A_targets_all_pairs：', A_targets_pairs)
    print('B_targets_all_pairs：', B_targets_pairs)

    save_pairs_file(epoch_name, A_holes_pairs,end='A',hot="hole") ##涉及函数4
    save_pairs_file(epoch_name, B_holes_pairs,end='B',hot="hole")
    save_pairs_file(epoch_name, A_targets_pairs,end='A',hot="target")
    save_pairs_file(epoch_name, B_targets_pairs,end='B',hot="target")

##此函数作用为计算三维点
def  epoch_3Dpoints(epoch_name): ##主要函数2
    """
    本函数主要处理步骤
    1、拿到前面所有的处理文件
    包括左右相机内参矩阵；左右相机R、T关系；匹配点数据
    2、确定最左点，去除畸变(具体还没看懂)
    3、将参数传入xy2xyz计算
    """
    data_root_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/')
    pairs_3D_path = os.path.join(data_root_path,epoch_name+'/points_info.xml')  ##匹配点与三维点保存在同一个文件

    dom = minidom.parse(pairs_3D_path)  ##points_info.xml文件解析点
    root = dom.documentElement

    AH_numsOfPairs = len(root.getElementsByTagName("pairs")[0].childNodes)
    AT_numsOfPairs = len(root.getElementsByTagName("pairs")[1].childNodes)
    BH_numsOfPairs = len(root.getElementsByTagName("pairs")[2].childNodes)
    BT_numsOfPairs = len(root.getElementsByTagName("pairs")[3].childNodes)
    ## 清除原始数据
    theNode0 = root.getElementsByTagName("hole")[0]
    theNode0.removeChild(root.getElementsByTagName("threeD")[0])
    theNode0.appendChild(dom.createElement("threeD"))
    theNode1 = root.getElementsByTagName("target")[0]
    theNode1.removeChild(root.getElementsByTagName("threeD")[1])
    theNode1.appendChild(dom.createElement("threeD"))
    theNode2 = root.getElementsByTagName("hole")[1]
    theNode2.removeChild(root.getElementsByTagName("threeD")[2])
    theNode2.appendChild(dom.createElement("threeD"))
    theNode3 = root.getElementsByTagName("target")[1]
    theNode3.removeChild(root.getElementsByTagName("threeD")[3])
    theNode3.appendChild(dom.createElement("threeD"))
    ## 计算A端孔的三维坐标并存入文件
    threeD = root.getElementsByTagName("threeD")[0]
    for i in range(AH_numsOfPairs):
        td = dom.createElement("point"+str(i))
        left_point = [0,0]
        right_point = [0,0]
        left_point[0] = root.getElementsByTagName("left_x")[i].childNodes[0].data
        left_point[1] = root.getElementsByTagName("left_y")[i].childNodes[0].data
        right_point[0] = root.getElementsByTagName("right_x")[i].childNodes[0].data
        right_point[1] = root.getElementsByTagName("right_y")[i].childNodes[0].data
        print(left_point)
        print(right_point)
        radius = root.getElementsByTagName("radius")[i].childNodes[0].data
        score = root.getElementsByTagName("score")[i].childNodes[0].data
        L = point_undistort(left_point,"left",end='A')  ## 函数增加端面参数
        R = point_undistort(right_point,"right",end='A')
        xyz = xy2xyz1(L,R,end='A') ## 函数增加端面参数
        print(xyz)
        X = dom.createElement("X")
        X.appendChild(dom.createTextNode(str(xyz[0][0])))
        td.appendChild(X)
        Y = dom.createElement("Y")
        Y.appendChild(dom.createTextNode(str(xyz[1][0])))
        td.appendChild(Y)
        Z = dom.createElement("Z")
        Z.appendChild(dom.createTextNode(str(xyz[2][0])))
        td.appendChild(Z)
        R = dom.createElement("R")
        R.appendChild(dom.createTextNode(radius))
        td.appendChild(R)
        S = dom.createElement("S")
        S.appendChild(dom.createTextNode(score))
        td.appendChild(S)
        threeD.appendChild(td)
    ## 计算A端标的三维坐标
    threeD = root.getElementsByTagName("threeD")[1]
    for i in range(AT_numsOfPairs):
        td = dom.createElement("point"+str(i))
        left_point = [0,0]
        right_point = [0,0]
        left_point[0] = root.getElementsByTagName("left_x")[i+AH_numsOfPairs].childNodes[0].data
        left_point[1] = root.getElementsByTagName("left_y")[i+AH_numsOfPairs].childNodes[0].data
        right_point[0] = root.getElementsByTagName("right_x")[i+AH_numsOfPairs].childNodes[0].data
        right_point[1] = root.getElementsByTagName("right_y")[i+AH_numsOfPairs].childNodes[0].data
        print(left_point)
        print(right_point)
        radius = root.getElementsByTagName("radius")[i+AH_numsOfPairs].childNodes[0].data
        score = root.getElementsByTagName("score")[i+AH_numsOfPairs].childNodes[0].data
        L = point_undistort(left_point,"left",end='A')
        R = point_undistort(right_point,"right",end='A')
        xyz = xy2xyz1(L,R,end='A')
        print(xyz)
        X = dom.createElement("X")
        X.appendChild(dom.createTextNode(str(xyz[0][0])))
        td.appendChild(X)
        Y = dom.createElement("Y")
        Y.appendChild(dom.createTextNode(str(xyz[1][0])))
        td.appendChild(Y)
        Z = dom.createElement("Z")
        Z.appendChild(dom.createTextNode(str(xyz[2][0])))
        td.appendChild(Z)
        R = dom.createElement("R")
        R.appendChild(dom.createTextNode(radius))
        td.appendChild(R)
        S = dom.createElement("S")
        S.appendChild(dom.createTextNode(score))
        td.appendChild(S)
        threeD.appendChild(td)
    ## 计算B端孔的三维坐标
    threeD = root.getElementsByTagName("threeD")[2]
    for i in range(BH_numsOfPairs):
        td = dom.createElement("point"+str(i))
        left_point = [0, 0]
        right_point = [0, 0]
        left_point[0] = root.getElementsByTagName("left_x")[i + AH_numsOfPairs+AT_numsOfPairs].childNodes[0].data
        left_point[1] = root.getElementsByTagName("left_y")[i + AH_numsOfPairs+AT_numsOfPairs].childNodes[0].data
        right_point[0] = root.getElementsByTagName("right_x")[i + AH_numsOfPairs+AT_numsOfPairs].childNodes[0].data
        right_point[1] = root.getElementsByTagName("right_y")[i + AH_numsOfPairs+AT_numsOfPairs].childNodes[0].data
        print(left_point)
        print(right_point)
        radius = root.getElementsByTagName("radius")[i + AH_numsOfPairs+AT_numsOfPairs].childNodes[0].data
        score = root.getElementsByTagName("score")[i + AH_numsOfPairs+AT_numsOfPairs].childNodes[0].data
        L = point_undistort(left_point,"left",end='B')
        R = point_undistort(right_point, "right",end='B')
        xyz = xy2xyz1(L, R,end='B')
        print(xyz)
        X = dom.createElement("X")
        X.appendChild(dom.createTextNode(str(xyz[0][0])))
        td.appendChild(X)
        Y = dom.createElement("Y")
        Y.appendChild(dom.createTextNode(str(xyz[1][0])))
        td.appendChild(Y)
        Z = dom.createElement("Z")
        Z.appendChild(dom.createTextNode(str(xyz[2][0])))
        td.appendChild(Z)
        R = dom.createElement("R")
        R.appendChild(dom.createTextNode(radius))
        td.appendChild(R)
        S = dom.createElement("S")
        S.appendChild(dom.createTextNode(score))
        td.appendChild(S)
        threeD.appendChild(td)
    ## 计算B端标的三维坐标
    threeD = root.getElementsByTagName("threeD")[3]
    for i in range(BT_numsOfPairs):
        td = dom.createElement("point"+str(i))
        left_point = [0, 0]
        right_point = [0, 0]
        left_point[0] = root.getElementsByTagName("left_x")[i + AH_numsOfPairs + AT_numsOfPairs+BH_numsOfPairs].childNodes[0].data
        left_point[1] = root.getElementsByTagName("left_y")[i + AH_numsOfPairs + AT_numsOfPairs+BH_numsOfPairs].childNodes[0].data
        right_point[0] = root.getElementsByTagName("right_x")[i + AH_numsOfPairs + AT_numsOfPairs+BH_numsOfPairs].childNodes[0].data
        right_point[1] = root.getElementsByTagName("right_y")[i + AH_numsOfPairs + AT_numsOfPairs+BH_numsOfPairs].childNodes[0].data
        print(left_point)
        print(right_point)
        radius = root.getElementsByTagName("radius")[i + AH_numsOfPairs + AT_numsOfPairs+BH_numsOfPairs].childNodes[0].data
        score = root.getElementsByTagName("score")[i + AH_numsOfPairs + AT_numsOfPairs+BH_numsOfPairs].childNodes[0].data
        L = point_undistort(left_point, "left",end='B')
        R = point_undistort(right_point, "right",end='B')
        xyz = xy2xyz1(L, R,end='B')
        print(xyz)
        X = dom.createElement("X")
        X.appendChild(dom.createTextNode(str(xyz[0][0])))
        td.appendChild(X)
        Y = dom.createElement("Y")
        Y.appendChild(dom.createTextNode(str(xyz[1][0])))
        td.appendChild(Y)
        Z = dom.createElement("Z")
        Z.appendChild(dom.createTextNode(str(xyz[2][0])))
        td.appendChild(Z)
        R = dom.createElement("R")
        R.appendChild(dom.createTextNode(radius))
        td.appendChild(R)
        S = dom.createElement("S")
        S.appendChild(dom.createTextNode(score))
        td.appendChild(S)
        threeD.appendChild(td)

    # for i,item in enumerate(data):
    #     print(item)
    #     left_point =  item['left_point']
    #     right_point = item['right_point']
    #
    #     L = point_undistort(left_point, 'left') ##涉及函数6
    #     R = point_undistort(right_point, 'right')
    #
    #     xyz = xy2xyz1(L,R,mLeftIntrinsic,mRightIntrinsic,R_matrix,T_matrix)
    #     xyz = [int(xyz[0])-xyz_flag[0],int(xyz[1])-xyz_flag[1],int(xyz[2])-xyz_flag[2]]
    #     print(xyz,xyz_flag)
    #     xyz_show = str(str(int(xyz[0]))+','+str(int(xyz[1]))+','+str(int(xyz[2])))
    #
    #     points_3d[str(int(left_point[0]))+','+str(int(left_point[1]))]= xyz
    #
    #     points_3d[str(int(right_point[0])) + ',' + str(int(right_point[1]))] = xyz
    with open(pairs_3D_path,'w') as fp:
        dom.writexml(fp)








def  epoch_3Dpoints_show(epoch_name): ##主要函数3


    root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeRestruction',
                                                                    'static/zj_pictures/result/points_3d')

    points3d_path = os.path.join(root_path, epoch_name, 'points_3d.json')

    points_3dRestruction_path = '/static/zj_pictures/points_3dRestruction'

    points_3dRestruction_path = os.path.join(points_3dRestruction_path,epoch_name)

    left_pic_path = os.path.join(points_3dRestruction_path,'left.jpg')
    right_pic_path = os.path.join(points_3dRestruction_path,'right.jpg')

    flag_path = os.path.dirname(os.path.realpath(__file__)).replace('bracketSearch','')

    left_pic_path =left_pic_path.replace(flag_path,'')

    right_pic_path = right_pic_path.replace(flag_path, '')

    info = {}

    info['points3d_path'] = points3d_path
    info['left_pic_path'] = left_pic_path
    info['right_pic_path'] = right_pic_path

    return  info



#####################################################################


if __name__ == '__main__':
    epoch_name = "2021-01-16-16:03:26"
    epoch_3Dpoints(epoch_name)
