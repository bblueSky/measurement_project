#encoding:utf-8

from flask  import render_template, request
from stereo_vision.resultsAnalysis import  resultsAnalysis
import time
import json
import cv2
import numpy as np
import time
import datetime
import signal
import threading
import os
import stereo_vision.ksj.cam as  kcam

from  stereo_vision.cameraCalibration.utils  import  sig_calibration,stereo_Calibration



@resultsAnalysis.route('/')
def index():

    return render_template('results_analysis/results_analysis.html')


@resultsAnalysis.route('/single/',methods=['POST','GET'])
def  single_calibration():


     path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'checkboard_img_dir/leftCamera_img_dir/')

     sig_calibration(path,camera_='left')

     path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'checkboard_img_dir/rightCamera_img_dir/')

     print('开始')
     sig_calibration(path,camera_='right')

     return  str(1)


@resultsAnalysis.route('/distortion',methods=['POST','GET'])
def  distortion_correct():

    flag = request.args.get('mydata')
    print(flag)

    #  1 是 拍摄状态下
    if  flag =='1' :

        img_path ="/home/cx/PycharmProjects/stereo_vision/stereo_vision/bracket_img_dir/"

    # 0 是 标定状态下
    elif  flag =='0':

        img_path ="/home/cx/PycharmProjects/stereo_vision/stereo_vision/checkboard_img_dir"

    img = "123"
    #return render_template('camera_calibration/calibration_output.html',img_path=img)
    return img


@resultsAnalysis.route('/stereo/',methods=['POST','GET'])
def  stereo_calibration():


    stereo_Calibration()

    return str(1)


@resultsAnalysis.route('/take_pic/', methods =['POST','GET'] )
def   output():
    print('------------------------------------------------------------------------------------')
    flag = request.get_data('mydata')
    print(flag)
    #os.system('sh /home/cx/PycharmProjects/stereo_vision/stereo_vision/cameraCalibration/run.sh')
    exp_time = 100
    #exp_time =  request.get_date('exp_time')
    cam = kcam.Ksjcam()

    #TODO 软触发模式设置

    cam.SetExptime(0, exp_time)  # 设置曝光150MS
    cam.SetTriggerMode(0, 2)  # 设置成固定帧率模式

    cam.SetExptime(1, exp_time)  # 设置曝光150MS
    cam.SetTriggerMode(1, 2)  # 设置成固定帧率模式

    #cam.SetFixedFrameRateEx(0, 1)  # 设置固定帧率是5fs

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')

    image_left = cam.read(0)  # 从相机0读取一个图像，这个image就是oenpcv的图像  # todo  需要找出哪个是相机0 哪个是相机1
    image_right = cam.read(1)
    frame_left = image_left
    frame_right = image_right

    # TODO  注释掉写入帧的操作
    dirPath = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/zj_pictures/temp/')

    flag = "calibration"
    if  flag =="calibration":

        #TODO  set checkboard_pic path

        img_left_save_path  = os.path.join(dirPath,'left.jpg')
        img_right_save_path = os.path.join(dirPath,'right.jpg')
        #img_right_save_path = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/temp/right.jpg'

    elif  flag =='take_pic':

        img_left_save_path = os.path.join(dirPath, 'left.jpg')
        img_right_save_path = os.path.join(dirPath, 'right.jpg')

        #img_left_save_path  = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/temp/left.jpg'
        #img_right_save_path = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/temp/right.jpg'


    cv2.imwrite(img_left_save_path, frame_left)
    cv2.imwrite(img_right_save_path, frame_right)

    return  "123"

@resultsAnalysis.route('/delete_pic/', methods =['POST','GET'] )
def  delete_pic():

    dirPath = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/zj_pictures/temp/')

    #dirPath = "/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/temp/"

    pic_file = os.listdir(dirPath)

    print('移除前test目录下有文件：%s' % os.listdir(dirPath))
    # 判断文件是否存在
    flag_dic={}
    if  len(pic_file)!=0:

        flag_dic["exist"] = 1
        for  item in  pic_file:
            os.remove(dirPath + item)
    else:

        flag_dic["exist"] = 2
        print("要删除的文件不存在！")

    print('----',json.dumps(flag_dic))
    return json.dumps(flag_dic)


@resultsAnalysis.route('/save_pic/', methods =['POST','GET'] )
def  save_pic():

    flag = request.args.get('mydata')

    left_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/zj_pictures/temp/left.jpg')
    right_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/zj_pictures/temp/right.jpg')
    left = cv2.imread(left_path)
    right= cv2.imread(right_path)

    #  1 是 拍摄状态下
    if flag == '1':

        left_camere_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/zj_pictures/left/')
        right_camere_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/zj_pictures/right/')

        file_dir = time.strftime("%Y-%m-%d-%X", time.localtime())
        if not os.path.exists(os.path.join(left_camere_dir, file_dir)):
            left_path = os.path.join(left_camere_dir, file_dir)
            os.makedirs(left_path)
            right_path = os.path.join(right_camere_dir, file_dir)
            os.makedirs(right_path)
            cv2.imwrite(os.path.join(left_path, file_dir+'_'+'left.jpg'), left)
            cv2.imwrite(os.path.join(right_path, file_dir+'_'+'right.jpg'), right)

    # 0 是 标定状态下
    elif flag == '0':

        left_camere_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
                                                                              'checkboard_img_dir/leftCamera_img_dir/')
        right_camere_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
                                                                               'checkboard_img_dir/rightCamera_img_dir/')

        file_name = time.strftime("%Y-%m-%d-%X", time.localtime())
        cv2.imwrite(os.path.join(left_camere_dir,file_name+'_'+'left.jpg'),left)
        cv2.imwrite(os.path.join(right_camere_dir,file_name+'_'+'right.jpg'), right)

    return flag









