#encoding:utf-8

from flask  import render_template, request
from stereo_vision.cameraCalibration import  cameraCalibration
import json
import cv2
import numpy as np
import time
import datetime
import signal
import threading
import os
from xml.dom import minidom
import stereo_vision.ksj.cam as  kcam

from  stereo_vision.cameraCalibration.utils  import  sig_calibration,stereo_Calibration
from  stereo_vision.cameraCalibration.utils  import  LTOrd2AOrd,LTside2VSide  ##还没写

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
    flag = request.args.get('mydata')
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

    cam.SetExptime(2, exp_time)  # 设置曝光150MS
    cam.SetTriggerMode(2, 2)  # 设置成固定帧率模式

    cam.SetExptime(3, exp_time)  # 设置曝光150MS
    cam.SetTriggerMode(3, 2)  # 设置成固定帧率模式

    #cam.SetFixedFrameRateEx(0, 1)  # 设置固定帧率是5fs

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime( '%Y'+ '-' + '%m' + '-' + '%d' + '-' + '%H' + ':' + '%M' + ':' + '%S')
    ##默认A端0\1 B端2\3
    if flag=='A':
        image_left = cam.read(0)  # 从相机0读取一个图像，这个image就是oenpcv的图像  # todo  需要找出哪个是相机0 哪个是相机1
        image_right = cam.read(1)
    else:
        image_left = cam.read(2)  # 从相机0读取一个图像，这个image就是oenpcv的图像  # todo  需要找出哪个是相机2 哪个是相机3
        image_right = cam.read(3)
    frame_left = image_left[:,::-1]
    frame_right = image_right[:,::-1]

    # TODO  注释掉写入帧的操作
    dirPath = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'static/res_pictures/temp/')

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
    res = dict()
    flag = request.args.get('mydata')
    resultl = sig_calibration(flag,"left")
    resultr = sig_calibration(flag,"right")
    results = stereo_Calibration(flag)

    res["ltime"] = resultl[0]
    res["lmtx0"] = resultl[1]
    res["lmtx1"] = resultl[2]
    res["lmtx2"] = resultl[3]
    res["ldist"] = resultl[4]
    res["lterror"] = resultl[5]

    res["rtime"] = resultr[0]
    res["rmtx0"] = resultr[1]
    res["rmtx1"] = resultr[2]
    res["rmtx2"] = resultr[3]
    res["rdist"] = resultr[4]
    res["rterror"] = resultr[5]

    res["R0"] = results[0]
    res["R1"] = results[1]
    res["R2"] = results[2]
    res["T"] = results[3]
    res["stime"] = results[4]

    # left_data = [50, 200, 360, 100, 100, 200]

    # right_data = [5, 20, 36, 10, 10, 20]

    res["left_data"] = resultl[6]
    res["right_data"] = resultr[6]

    return res


@cameraCalibration.route('/inputComplete/',methods=['POST','GET'])
def inputComplete():
    result = dict()
    filePath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","exteriorFile/LT.txt")
    res = list()
    with open(filePath, "r") as f:
        for i,line in enumerate(f.readlines()):
            if i>=6:
                break
            line = line.strip("\n")
            x,y,z = line[1:-1].split(',')
            res.append([x,y,z])
    # print(res)
    result["A1X"] = A1X = res[0][0]
    result["A1Y"] = A1Y = res[0][1]
    result["A1Z"] = A1Z = res[0][2]
    result["A2X"] = A2X = res[1][0]
    result["A2Y"] = A2Y = res[1][1]
    result["A2Z"] = A2Z = res[1][2]
    result["A3X"] = A3X = res[2][0]
    result["A3Y"] = A3Y = res[2][1]
    result["A3Z"] = A3Z = res[2][2]
    result["B1X"] = B1X = res[3][0]
    result["B1Y"] = B1Y = res[3][1]
    result["B1Z"] = B1Z = res[3][2]
    result["B2X"] = B2X = res[4][0]
    result["B2Y"] = B2Y = res[4][1]
    result["B2Z"] = B2Z = res[4][2]
    result["B3X"] = B3X = res[5][0]
    result["B3Y"] = B3Y = res[5][1]
    result["B3Z"] = B3Z = res[5][2]
    AP1 = np.mat([float(A1X), float(A1Y), float(A1Z)])
    AP2 = np.mat([float(A2X), float(A2Y), float(A2Z)])
    AP3 = np.mat([float(A3X), float(A3Y), float(A3Z)])
    BP1 = np.mat([float(B1X), float(B1Y), float(B1Z)])
    BP2 = np.mat([float(B2X), float(B2Y), float(B2Z)])
    BP3 = np.mat([float(B3X), float(B3Y), float(B3Z)])
    APO, APH, APW, BPO, BPH, BPW, R_T2A, T_T2A = LTOrd2AOrd(AP1, AP2, AP3, BP1, BP2,
                                                            BP3)  ##激光跟踪仪下的六点坐标转移到A基准板坐标系下；APO默认（0,0,0），APH与APO同Height，APW与APO同Width
    AOX = APO[0, 0]
    AOY = APO[0, 1]
    AOZ = APO[0, 2]
    result["APO"] = [AOX, AOY, AOZ]
    AHX = APH[0, 0]
    AHY = APH[0, 1]
    AHZ = APH[0, 2]
    result["APH"] = [AHX, AHY, AHZ]
    AWX = APW[0, 0]
    AWY = APW[0, 1]
    AWZ = APW[0, 2]
    result["APW"] = [AWX, AWY, AWZ]
    BOX = BPO[0, 0]
    BOY = BPO[0, 1]
    BOZ = BPO[0, 2]
    result["BPO"] = [BOX, BOY, BOZ]
    BHX = BPH[0, 0]
    BHY = BPH[0, 1]
    BHZ = BPH[0, 2]
    result["BPH"] = [BHX, BHY, BHZ]
    BWX = BPW[0, 0]
    BWY = BPW[0, 1]
    BWZ = BPW[0, 2]
    result["BPW"] = [BWX, BWY, BWZ]
    print("LT转至A板的R为:===============\n")
    print(R_T2A)
    print("LT转至A板的T为:===============\n")
    print(T_T2A)
    APOs, APHs, APWs, BPOs, BPHs, BPWs, T_AS2S, T_BS2S = LTside2VSide(APO, APH, APW, BPO, BPH,
                                                                      BPW)  ##靶球一侧向视觉靶标一侧转换，坐标系仍然是A基准板，解出的六点是视觉靶标在A板坐标系下的坐标
    AOsX = APOs[0, 0]
    AOsY = APOs[0, 1]
    AOsZ = APOs[0, 2]
    result["APOs"] = [AOsX, AOsY, AOsZ]
    AHsX = APHs[0, 0]
    AHsY = APHs[0, 1]
    AHsZ = APHs[0, 2]
    result["APHs"] = [AHsX, AHsY, AHsZ]
    AWsX = APWs[0, 0]
    AWsY = APWs[0, 1]
    AWsZ = APWs[0, 2]
    result["APWs"] = [AWsX, AWsY, AWsZ]
    BOsX = BPOs[0, 0]
    BOsY = BPOs[0, 1]
    BOsZ = BPOs[0, 2]
    result["BPOs"] = [BOsX, BOsY, BOsZ]
    BHsX = BPHs[0, 0]
    BHsY = BPHs[0, 1]
    BHsZ = BPHs[0, 2]
    result["BPHs"] = [BHsX, BHsY, BHsZ]
    BWsX = BPWs[0, 0]
    BWsY = BPWs[0, 1]
    BWsZ = BPWs[0, 2]
    result["BPWs"] = [BWsX, BWsY, BWsZ]
    ATX = T_AS2S[0, 0]
    ATY = T_AS2S[0, 1]
    ATZ = T_AS2S[0, 2]
    BTX = T_BS2S[0, 0]
    BTY = T_BS2S[0, 1]
    BTZ = T_BS2S[0, 2]
    print("A板一侧转至另一侧的T为:=========\n")
    print(T_AS2S)
    print("A板一侧转至另一侧的T为:=========\n")
    print(T_BS2S)
    ## 数据存入global_ord.xml
    savePath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration", "static/global_info.xml")
    dom = minidom.parse(savePath)
    root = dom.documentElement
    root.removeChild(root.getElementsByTagName("origin")[0])
    root.removeChild(root.getElementsByTagName("T2A")[0])
    root.removeChild(root.getElementsByTagName("S2S")[0])
    root.appendChild(dom.createElement("origin"))
    root.appendChild(dom.createElement("T2A"))
    root.appendChild(dom.createElement("S2S"))
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime(
        '%Y' + '-' + '%m' + '-' + '%d' + '-' + '%H' + ':' + '%M' + ':' + '%S')
    calibration_time = root.getElementsByTagName("global_time")[0]
    calibration_time.removeChild(root.getElementsByTagName("time")[2])
    time1 = dom.createElement("time")
    calibration_time.appendChild(time1)
    time1.appendChild(dom.createTextNode(st))
    origin = root.getElementsByTagName("origin")[0]
    AP1 = dom.createElement('AP1')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(A1X))
    AP1.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(A1Y))
    AP1.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(A1Z))
    AP1.appendChild(Z)
    AP2 = dom.createElement('AP2')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(A2X))
    AP2.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(A2Y))
    AP2.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(A2Z))
    AP2.appendChild(Z)
    AP3 = dom.createElement('AP3')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(A3X))
    AP3.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(A3Y))
    AP3.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(A3Z))
    AP3.appendChild(Z)
    BP1 = dom.createElement('BP1')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(B1X))
    BP1.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(B1Y))
    BP1.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(B1Z))
    BP1.appendChild(Z)
    BP2 = dom.createElement('BP2')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(B2X))
    BP2.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(B2Y))
    BP2.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(B2Z))
    BP2.appendChild(Z)
    BP3 = dom.createElement('BP3')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(B3X))
    BP3.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(B3Y))
    BP3.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(B3Z))
    BP3.appendChild(Z)
    origin.appendChild(AP1)
    origin.appendChild(AP2)
    origin.appendChild(AP3)
    origin.appendChild(BP1)
    origin.appendChild(BP2)
    origin.appendChild(BP3)

    T2A = root.getElementsByTagName("T2A")[0]
    APO = dom.createElement('APO')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AOX)))
    APO.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AOY)))
    APO.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AOZ)))
    APO.appendChild(Z)
    APH = dom.createElement('APH')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AHX)))
    APH.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AHY)))
    APH.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AHZ)))
    APH.appendChild(Z)
    APW = dom.createElement('APW')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AWX)))
    APW.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AWY)))
    APW.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AWZ)))
    APW.appendChild(Z)
    BPO = dom.createElement('BPO')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BOX)))
    BPO.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BOY)))
    BPO.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BOZ)))
    BPO.appendChild(Z)
    BPH = dom.createElement('BPH')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BHX)))
    BPH.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BHY)))
    BPH.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BHZ)))
    BPH.appendChild(Z)
    BPW = dom.createElement('BPW')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BWX)))
    BPW.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BWY)))
    BPW.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BWZ)))
    BPW.appendChild(Z)
    T2A.appendChild(APO)
    T2A.appendChild(APH)
    T2A.appendChild(APW)
    T2A.appendChild(BPO)
    T2A.appendChild(BPH)
    T2A.appendChild(BPW)

    S2S = root.getElementsByTagName("S2S")[0]
    APOs = dom.createElement('APOs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AOsX)))
    APOs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AOsY)))
    APOs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AOsZ)))
    APOs.appendChild(Z)
    APHs = dom.createElement('APHs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AHsX)))
    APHs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AHsY)))
    APHs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AHsZ)))
    APHs.appendChild(Z)
    APWs = dom.createElement('APWs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AWsX)))
    APWs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AWsY)))
    APWs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AWsZ)))
    APWs.appendChild(Z)
    BPOs = dom.createElement('BPOs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BOsX)))
    BPOs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BOsY)))
    BPOs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BOsZ)))
    BPOs.appendChild(Z)
    BPHs = dom.createElement('BPHs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BHsX)))
    BPHs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BHsY)))
    BPHs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BHsZ)))
    BPHs.appendChild(Z)
    BPWs = dom.createElement('BPWs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BWsX)))
    BPWs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BWsY)))
    BPWs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BWsZ)))
    BPWs.appendChild(Z)
    T_AS2S = dom.createElement('T_AS2S')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(ATX)))
    T_AS2S.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(ATY)))
    T_AS2S.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(ATZ)))
    T_AS2S.appendChild(Z)
    T_BS2S = dom.createElement('T_BS2S')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BTX)))
    T_BS2S.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BTY)))
    T_BS2S.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BTZ)))
    T_BS2S.appendChild(Z)
    S2S.appendChild(T_AS2S)
    S2S.appendChild(T_BS2S)
    S2S.appendChild(APOs)
    S2S.appendChild(APHs)
    S2S.appendChild(APWs)
    S2S.appendChild(BPOs)
    S2S.appendChild(BPHs)
    S2S.appendChild(BPWs)
    with open(savePath, 'w') as fp:
        dom.writexml(fp)

    return result


@cameraCalibration.route('/insertComplete/',methods=['POST','GET'])
def insertComplete():
    A1X = request.args.get('A1X')
    A1Y = request.args.get('A1Y')
    A1Z = request.args.get('A1Z')
    A2X = request.args.get('A2X')
    A2Y = request.args.get('A2Y')
    A2Z = request.args.get('A2Z')
    A3X = request.args.get('A3X')
    A3Y = request.args.get('A3Y')
    A3Z = request.args.get('A3Z')
    B1X = request.args.get('B1X')
    B1Y = request.args.get('B1Y')
    B1Z = request.args.get('B1Z')
    B2X = request.args.get('B2X')
    B2Y = request.args.get('B2Y')
    B2Z = request.args.get('B2Z')
    B3X = request.args.get('B3X')
    B3Y = request.args.get('B3Y')
    B3Z = request.args.get('B3Z')
    AP1 = np.mat([float(A1X),float(A1Y),float(A1Z)])
    AP2 = np.mat([float(A2X),float(A2Y),float(A2Z)])
    AP3 = np.mat([float(A3X),float(A3Y),float(A3Z)])
    BP1 = np.mat([float(B1X),float(B1Y),float(B1Z)])
    BP2 = np.mat([float(B2X),float(B2Y),float(B2Z)])
    BP3 = np.mat([float(B3X),float(B3Y),float(B3Z)])
    APO,APH,APW,BPO,BPH,BPW,R_T2A,T_T2A = LTOrd2AOrd(AP1,AP2,AP3,BP1,BP2,BP3)  ##激光跟踪仪下的六点坐标转移到A基准板坐标系下；APO默认（0,0,0），APH与APO同Height，APW与APO同Width
    result = dict()
    AOX = APO[0, 0]
    AOY = APO[0, 1]
    AOZ = APO[0, 2]
    result["APO"] = [AOX,AOY,AOZ]
    AHX = APH[0, 0]
    AHY = APH[0, 1]
    AHZ = APH[0, 2]
    result["APH"] = [AHX,AHY,AHZ]
    AWX = APW[0, 0]
    AWY = APW[0, 1]
    AWZ = APW[0, 2]
    result["APW"] = [AWX,AWY,AWZ]
    BOX = BPO[0, 0]
    BOY = BPO[0, 1]
    BOZ = BPO[0, 2]
    result["BPO"] = [BOX,BOY,BOZ]
    BHX = BPH[0, 0]
    BHY = BPH[0, 1]
    BHZ = BPH[0, 2]
    result["BPH"] = [BHX,BHY,BHZ]
    BWX = BPW[0, 0]
    BWY = BPW[0, 1]
    BWZ = BPW[0, 2]
    result["BPW"] = [BWX,BWY,BWZ]
    print("LT转至A板的R为:===============\n")
    print(R_T2A)
    print("LT转至A板的T为:===============\n")
    print(T_T2A)
    APOs,APHs,APWs,BPOs,BPHs,BPWs,T_AS2S,T_BS2S = LTside2VSide(APO,APH,APW,BPO,BPH,BPW)  ##靶球一侧向视觉靶标一侧转换，坐标系仍然是A基准板，解出的六点是视觉靶标在A板坐标系下的坐标
    AOsX = APOs[0, 0]
    AOsY = APOs[0, 1]
    AOsZ = APOs[0, 2]
    result["APOs"] = [AOsX,AOsY,AOsZ]
    AHsX = APHs[0, 0]
    AHsY = APHs[0, 1]
    AHsZ = APHs[0, 2]
    result["APHs"] = [AHsX,AHsY,AHsZ]
    AWsX = APWs[0, 0]
    AWsY = APWs[0, 1]
    AWsZ = APWs[0, 2]
    result["APWs"] = [AWsX,AWsY,AWsZ]
    BOsX = BPOs[0, 0]
    BOsY = BPOs[0, 1]
    BOsZ = BPOs[0, 2]
    result["BPOs"] = [BOsX,BOsY,BOsZ]
    BHsX = BPHs[0, 0]
    BHsY = BPHs[0, 1]
    BHsZ = BPHs[0, 2]
    result["BPHs"] = [BHsX,BHsY,BHsZ]
    BWsX = BPWs[0, 0]
    BWsY = BPWs[0, 1]
    BWsZ = BPWs[0, 2]
    result["BPWs"] = [BWsX,BWsY,BWsZ]
    ATX = T_AS2S[0, 0]
    ATY = T_AS2S[0, 1]
    ATZ = T_AS2S[0, 2]
    BTX = T_BS2S[0, 0]
    BTY = T_BS2S[0, 1]
    BTZ = T_BS2S[0, 2]
    print("A板一侧转至另一侧的T为:=========\n")
    print(T_AS2S)
    print("A板一侧转至另一侧的T为:=========\n")
    print(T_BS2S)
    ## 数据存入global_ord.xml
    savePath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/global_info.xml")
    dom = minidom.parse(savePath)
    root = dom.documentElement
    root.removeChild(root.getElementsByTagName("origin")[0])
    root.removeChild(root.getElementsByTagName("T2A")[0])
    root.removeChild(root.getElementsByTagName("S2S")[0])
    root.appendChild(dom.createElement("origin"))
    root.appendChild(dom.createElement("T2A"))
    root.appendChild(dom.createElement("S2S"))
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y' + '-' + '%m' + '-' + '%d' + '-' + '%H' + ':' + '%M' + ':' + '%S')
    calibration_time = root.getElementsByTagName("global_time")[0]
    calibration_time.removeChild(root.getElementsByTagName("time")[2])
    time1 = dom.createElement("time")
    calibration_time.appendChild(time1)
    time1.appendChild(dom.createTextNode(st))
    origin = root.getElementsByTagName("origin")[0]
    AP1 = dom.createElement('AP1')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(A1X))
    AP1.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(A1Y))
    AP1.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(A1Z))
    AP1.appendChild(Z)
    AP2 = dom.createElement('AP2')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(A2X))
    AP2.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(A2Y))
    AP2.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(A2Z))
    AP2.appendChild(Z)
    AP3 = dom.createElement('AP3')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(A3X))
    AP3.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(A3Y))
    AP3.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(A3Z))
    AP3.appendChild(Z)
    BP1 = dom.createElement('BP1')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(B1X))
    BP1.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(B1Y))
    BP1.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(B1Z))
    BP1.appendChild(Z)
    BP2 = dom.createElement('BP2')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(B2X))
    BP2.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(B2Y))
    BP2.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(B2Z))
    BP2.appendChild(Z)
    BP3 = dom.createElement('BP3')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(B3X))
    BP3.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(B3Y))
    BP3.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(B3Z))
    BP3.appendChild(Z)
    origin.appendChild(AP1)
    origin.appendChild(AP2)
    origin.appendChild(AP3)
    origin.appendChild(BP1)
    origin.appendChild(BP2)
    origin.appendChild(BP3)




    T2A = root.getElementsByTagName("T2A")[0]
    APO = dom.createElement('APO')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AOX)))
    APO.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AOY)))
    APO.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AOZ)))
    APO.appendChild(Z)
    APH = dom.createElement('APH')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AHX)))
    APH.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AHY)))
    APH.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AHZ)))
    APH.appendChild(Z)
    APW = dom.createElement('APW')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AWX)))
    APW.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AWY)))
    APW.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AWZ)))
    APW.appendChild(Z)
    BPO = dom.createElement('BPO')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BOX)))
    BPO.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BOY)))
    BPO.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BOZ)))
    BPO.appendChild(Z)
    BPH = dom.createElement('BPH')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BHX)))
    BPH.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BHY)))
    BPH.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BHZ)))
    BPH.appendChild(Z)
    BPW = dom.createElement('BPW')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BWX)))
    BPW.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BWY)))
    BPW.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BWZ)))
    BPW.appendChild(Z)
    T2A.appendChild(APO)
    T2A.appendChild(APH)
    T2A.appendChild(APW)
    T2A.appendChild(BPO)
    T2A.appendChild(BPH)
    T2A.appendChild(BPW)



    S2S = root.getElementsByTagName("S2S")[0]
    APOs = dom.createElement('APOs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AOsX)))
    APOs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AOsY)))
    APOs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AOsZ)))
    APOs.appendChild(Z)
    APHs = dom.createElement('APHs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AHsX)))
    APHs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AHsY)))
    APHs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AHsZ)))
    APHs.appendChild(Z)
    APWs = dom.createElement('APWs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(AWsX)))
    APWs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(AWsY)))
    APWs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(AWsZ)))
    APWs.appendChild(Z)
    BPOs = dom.createElement('BPOs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BOsX)))
    BPOs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BOsY)))
    BPOs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BOsZ)))
    BPOs.appendChild(Z)
    BPHs = dom.createElement('BPHs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BHsX)))
    BPHs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BHsY)))
    BPHs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BHsZ)))
    BPHs.appendChild(Z)
    BPWs = dom.createElement('BPWs')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BWsX)))
    BPWs.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BWsY)))
    BPWs.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BWsZ)))
    BPWs.appendChild(Z)
    T_AS2S = dom.createElement('T_AS2S')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(ATX)))
    T_AS2S.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(ATY)))
    T_AS2S.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(ATZ)))
    T_AS2S.appendChild(Z)
    T_BS2S = dom.createElement('T_BS2S')
    X = dom.createElement('X')
    X.appendChild(dom.createTextNode(str(BTX)))
    T_BS2S.appendChild(X)
    Y = dom.createElement('Y')
    Y.appendChild(dom.createTextNode(str(BTY)))
    T_BS2S.appendChild(Y)
    Z = dom.createElement('Z')
    Z.appendChild(dom.createTextNode(str(BTZ)))
    T_BS2S.appendChild(Z)
    S2S.appendChild(T_AS2S)
    S2S.appendChild(T_BS2S)
    S2S.appendChild(APOs)
    S2S.appendChild(APHs)
    S2S.appendChild(APWs)
    S2S.appendChild(BPOs)
    S2S.appendChild(BPHs)
    S2S.appendChild(BPWs)
    with open(savePath, 'w') as fp:
        dom.writexml(fp)

    return result


@cameraCalibration.route('/constructMeasureField/',methods=['POST','GET'])
def constructMeasureField():
    filePath = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/global_info.xml")
    doc = minidom.parse(filePath)
    root = doc.documentElement
    AcameraTime = root.getElementsByTagName("time")[0].childNodes[0].data
    BcameraTime = root.getElementsByTagName("time")[1].childNodes[0].data
    globalTime = root.getElementsByTagName("time")[2].childNodes[0].data
    res = dict()
    res["AcameraTime"] = AcameraTime
    res["BcameraTime"] = BcameraTime
    res["globalTime"] = globalTime
    return res





