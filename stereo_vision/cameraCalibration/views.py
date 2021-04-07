#encoding:utf-8

from flask  import render_template, request
from stereo_vision.cameraCalibration import  cameraCalibration
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


@cameraCalibration.route('/')
def index():

    return render_template('camera_calibration/camera_calibration.html')


@cameraCalibration.route('/selectAorB/',methods=['POST','GET'])
def  selectAorB():
    AorB = request.args.get('mydata')
    #print("传数成功，选择了"+flager)

    img_path = os.path.realpath(__file__).replace("cameraCalibration/views.py","static/checkboard_img_dir/"+AorB)
    return  str(1)


@cameraCalibration.route('/imageDisplay/',methods=['POST','GET'])
def  imageDisplay():

    flag = request.args.get('mydata')


    img = "123"
    #return render_template('camera_calibration/calibration_output.html',img_path=img)
    return img



@cameraCalibration.route('/take_pic/', methods =['POST','GET'] )
def take_pic():
    print("相机开始运行")
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
    st = datetime.datetime.fromtimestamp(ts).strftime( '%Y'+ '-' + '%m' + '-' + '%d' + '-' + '%H' + ':' + '%M' + ':' + '%S')

    image_left = cam.read(0)  # 从相机0读取一个图像，这个image就是oenpcv的图像  # todo  需要找出哪个是相机0 哪个是相机1
    image_right = cam.read(1)
    frame_left = image_left
    frame_right = image_right

    # TODO  注释掉写入帧的操作
    dirPath = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/res_pictures/temp/')
    flag = request.args.get('mydata')
    print("标定的是"+flag+"端相机")


    img_left_save_path = os.path.join(dirPath, 'left.jpg')
    img_right_save_path = os.path.join(dirPath, 'right.jpg')

    cv2.imwrite(img_left_save_path, frame_left)
    cv2.imwrite(img_right_save_path, frame_right)


    return  json.dumps({"key":st})

@cameraCalibration.route('/save_pic/',methods=['POST','GET'])
def save_pic():
    from_dirPath = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/res_pictures/temp/')
    to_dirPath = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/checkboard_img_dir/')
    flag = request.args.get('mydata')
    time = request.args.get('mydata1')
    from_img_left_path  = os.path.join(from_dirPath,'left.jpg')
    from_img_right_path = os.path.join(from_dirPath,'right.jpg')
    to_img_left_path = os.path.join(to_dirPath,flag+"_leftCamera_img_dir/"+time+"_left.jpg")
    to_img_right_path = os.path.join(to_dirPath,flag+"_rightCamera_img_dir/"+time+"_right.jpg")
    command1 = "cp "+from_img_left_path+" "+to_img_left_path
    command2 = "cp " + from_img_right_path + " " + to_img_right_path
    os.system(command1)
    os.system(command2)

    return str(1)



@cameraCalibration.route('/pageUp/',methods=['POST','GET'])
def pageUp():
    id = 0
    dirList = []
    flag = request.args.get('mydata')
    time = request.args.get('mydata1')
    left_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/checkboard_img_dir/'+flag+'_leftCamera_img_dir/')
    right_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/' + flag + '_rightCamera_img_dir/')
    dirList_f = os.listdir(left_path)
    for file in dirList_f:
        if os.path.splitext(file)[1] == '.jpg':
            dirList.append(file)
    num = len(dirList)
    cur_img = str(time)+"_left.jpg"
    #print(dirList)
    if cur_img not in dirList:
        cur_img = dirList[-1]
        id = num
    else:
        for i in range(num):
            if dirList[i] == cur_img:
                i+=-1
                if i <0:
                    cur_img = dirList[-1]
                    id = num
                    break
                else:
                    cur_img = dirList[i]
                    id = i+1
                    break
    time = cur_img[:-9]
    left_path = '/static/checkboard_img_dir/'+flag+'_leftCamera_img_dir/'+time+'_left.jpg'
    right_path = '/static/checkboard_img_dir/'+flag+'_rightCamera_img_dir/'+time+'_right.jpg'
    #print(left_path)
    #print(right_path)
    return {"left_path":left_path,"right_path":right_path,"time":time,"id":id,"num":num}




@cameraCalibration.route('/pageDown/',methods=['POST','GET'])
def pageDown():
    id = 0
    dirList = []
    flag = request.args.get('mydata')
    time = request.args.get('mydata1')
    left_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
                                                                    'static/checkboard_img_dir/' + flag + '_leftCamera_img_dir/')
    right_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
                                                                     'static/checkboard_img_dir/' + flag + '_rightCamera_img_dir/')
    dirList_f = os.listdir(left_path)
    for file in dirList_f:
        if os.path.splitext(file)[1] == '.jpg':
            dirList.append(file)
    num = len(dirList)
    cur_img = str(time) + "_left.jpg"
    # print(dirList)
    if cur_img not in dirList:
        cur_img = dirList[0]
        id = 1
    else:
        for i in range(num):
            if dirList[i] == cur_img:
                i += 1
                if i >num-1:
                    cur_img = dirList[0]
                    id = 1
                    break
                else:
                    cur_img = dirList[i]
                    id = i + 1
                    break
    time = cur_img[:-9]
    left_path = '/static/checkboard_img_dir/' + flag + '_leftCamera_img_dir/' + time + '_left.jpg'
    right_path = '/static/checkboard_img_dir/' + flag + '_rightCamera_img_dir/' + time + '_right.jpg'
    #print(left_path)
    #print(right_path)
    return {"left_path": left_path, "right_path": right_path, "time": time, "id": id, "num": num}


@cameraCalibration.route('/deleteThis/',methods=['POST','GET'])
def deletThis():
    dirList = []
    flag = request.args.get('mydata')
    time = request.args.get('mydata1')
    leftPath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/checkboard_img_dir/" + flag + "_leftCamera_img_dir/")
    rightPath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/checkboard_img_dir/" + flag + "_rightCamera_img_dir/")
    command1 = "rm -f "+leftPath+time+"_left.jpg"
    command2 = "rm -f "+rightPath+time+"_right.jpg"
    os.system(command1)
    os.system(command2)
    dirList_f = os.listdir(leftPath)
    for file in dirList_f:
        if os.path.splitext(file)[1] == '.jpg':
            dirList.append(file)
    num = len(dirList)
    # print(dirList)
    cur_img = dirList[-1]
    id = num
    time = cur_img[:-9]
    left_path = '/static/checkboard_img_dir/' + flag + '_leftCamera_img_dir/' + time + '_left.jpg'
    right_path = '/static/checkboard_img_dir/' + flag + '_rightCamera_img_dir/' + time + '_right.jpg'
    return {"left_path": left_path, "right_path": right_path, "time": time, "id": id, "num": num}


@cameraCalibration.route('/deleteAll/',methods=['POST','GET'])
def deletAll():
    flag = request.args.get('mydata')
    leftPath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/checkboard_img_dir/"+flag+"_leftCamera_img_dir/")
    rightPath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/checkboard_img_dir/" + flag + "_rightCamera_img_dir/")
    command1 = "rm -f "+leftPath+"*"
    command2 = "rm -f "+rightPath+"*"
    os.system(command1)
    os.system(command2)
    return str(1)


@cameraCalibration.route('/stereoCalibration/',methods=['POST','GET'])
def  stereo_calibration():
    flag = request.args.get('mydata')
    sig_calibration(flag,"left")
    sig_calibration(flag,"right")
    stereo_Calibration(flag)


    return str(1)


@cameraCalibration.route('/displayInfo/',methods=['POST','GET'])
def displayInfo():

    return str(1)


@cameraCalibration.route('/laserTracker/',methods=['POST','GET'])
def laserTracker():


    return str(1)


@cameraCalibration.route('/constructMeasureField/',methods=['POST','GET'])
def constructMeasureField():


    return str(1)





