#encoding:utf-8

from flask  import render_template, request
from stereo_vision.testMeasure import  testMeasure
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
from xml.dom import minidom

from  stereo_vision.testMeasure.utils  import  get_img_boxes
from stereo_vision.testMeasure.utils import img_process


@testMeasure.route('/')
def index():

    return render_template('test_measure/test_measure.html')


@testMeasure.route('/updateList1/',methods=['POST','GET'])
def  updateList1():
    #这里需要加检索筒端类型文件，把检索结果返回到字典里
    result = dict()
    type_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure","static/priori_data")
    type_list_f = os.listdir(type_path)
    type_list = list()
    for file in type_list_f:
        if os.path.splitext(file)[1] == '.xml':
            type_list.append(file)
    for i in range(1,len(type_list)+1):
        doc = minidom.parse(type_path+"/"+type_list[i-1])
        tube_name = doc.documentElement.getElementsByTagName("time")[0].childNodes[0].data
        result[i]=tube_name
    return result


@testMeasure.route('/selectType/',methods=['POST','GET'])
def  selectType():
    type = request.args.get('mydata')


    print(type)
    return  str(1)


@testMeasure.route('/imageDisplay/',methods=['POST','GET'])
def  imageDisplay():

    flag = request.args.get('mydata')


    img = "123"
    #return render_template('camera_calibration/calibration_output.html',img_path=img)
    return img



@testMeasure.route('/take_pic/', methods =['POST','GET'] )
def take_pic():
    img_path = os.path.realpath(__file__).replace("testMeasure/views.py", "static/res_pictures/")
    print("相机开始运行")
    #os.system('sh /home/cx/PycharmProjects/stereo_vision/stereo_vision/cameraCalibration/run.sh')
    exp_time = 70
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

    image_left_A = cam.read(0)  # 从相机0读取一个图像，这个image就是oenpcv的图像 先插的是相机1
    image_right_A = cam.read(1)
    image_left_B = image_left_A
    image_right_B = image_right_A
    #注意！！！这里以后要换成相机组！！！！暂时B端借用0\1
    # image_left_B = cam.read(2)
    # image_right_B = cam.read(3)

    frame_left_A = image_left_A[:,::-1]
    frame_right_A = image_right_A[:,::-1]
    frame_left_B = image_left_B[:,::-1]
    frame_right_B = image_right_B[:,::-1]
    # TODO  注释掉写入帧的操作
    dirPath = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/temp/')
    type = request.args.get('mydata')
    print("正在拍摄的是"+type+"型筒段")


    img_left_A_save_path = os.path.join(dirPath, 'left_A.jpg')
    img_right_A_save_path = os.path.join(dirPath, 'right_A.jpg')
    img_left_B_save_path = os.path.join(dirPath, 'left_B.jpg')
    img_right_B_save_path = os.path.join(dirPath, 'right_B.jpg')

    cv2.imwrite(img_left_A_save_path, frame_left_A)
    cv2.imwrite(img_right_A_save_path, frame_right_A)
    cv2.imwrite(img_left_B_save_path, frame_left_B)
    cv2.imwrite(img_right_B_save_path, frame_right_B)

    return  json.dumps({"key":st})

@testMeasure.route('/save_pic/',methods=['POST','GET'])
def save_pic():
    from_dirPath = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/temp/')
    to_dirPath = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/')
    type = request.args.get('mydata')
    type1 = request.args.get('mydata1')
    time = request.args.get('mydata2')
    from_img_left_A_path  = os.path.join(from_dirPath,'left_A.jpg')
    from_img_right_A_path = os.path.join(from_dirPath,'right_A.jpg')
    from_img_left_B_path = os.path.join(from_dirPath, 'left_B.jpg')
    from_img_right_B_path = os.path.join(from_dirPath, 'right_B.jpg')
    to_img_left_A_path = os.path.join(to_dirPath,"A_left/"+time+"_left.jpg")
    to_img_right_A_path = os.path.join(to_dirPath,"A_right/"+time+"_right.jpg")
    to_img_left_B_path = os.path.join(to_dirPath, "B_left/" + time + "_left.jpg")
    to_img_right_B_path = os.path.join(to_dirPath, "B_right/" + time + "_right.jpg")
    command1 = "cp "+from_img_left_A_path+" "+to_img_left_A_path
    command2 = "cp "+from_img_right_A_path+" "+to_img_right_A_path
    command3 = "cp "+from_img_left_B_path+" "+to_img_left_B_path
    command4 = "cp "+from_img_right_B_path+" "+to_img_right_B_path
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    result_path = to_dirPath+"result/"
    time_dirs_path = result_path+time
    os.mkdir(time_dirs_path)
    dir_list = ["A_left","A_right","B_left","B_right"]
    #在time_dirs_path下创建point_info.xml文件
    p_doc = minidom.Document()
    root = p_doc.createElement('points_info')
    root.setAttribute('数据时间',time)
    p_doc.appendChild(root)
    tube_type = p_doc.createElement('type')
    tube_type.appendChild(p_doc.createTextNode(type1))
    A_end = p_doc.createElement('A_end')
    A_hole = p_doc.createElement('hole')
    A_target = p_doc.createElement('target')
    A_target_pairs = p_doc.createElement('pairs')
    A_hole_pairs = p_doc.createElement('pairs')
    A_target_3D = p_doc.createElement('threeD')
    A_hole_3D = p_doc.createElement('threeD')
    A_hole.appendChild(A_hole_pairs)
    A_hole.appendChild(A_hole_3D)
    A_target.appendChild(A_target_pairs)
    A_target.appendChild(A_target_3D)
    A_end.appendChild(A_hole)
    A_end.appendChild(A_target)
    B_end = p_doc.createElement('B_end')
    B_hole = p_doc.createElement('hole')
    B_target = p_doc.createElement('target')
    B_target_pairs = p_doc.createElement('pairs')
    B_hole_pairs = p_doc.createElement('pairs')
    B_target_3D = p_doc.createElement('threeD')
    B_hole_3D = p_doc.createElement('threeD')
    B_hole.appendChild(B_hole_pairs)
    B_hole.appendChild(B_hole_3D)
    B_target.appendChild(B_target_pairs)
    B_target.appendChild(B_target_3D)
    B_end.appendChild(B_hole)
    B_end.appendChild(B_target)
    root.appendChild(tube_type)
    root.appendChild(A_end)
    root.appendChild(B_end)
    fp = open(time_dirs_path+'/points_info.xml','w')
    p_doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

    for i in dir_list:
        dir_path = time_dirs_path+"/"+i
        os.mkdir(dir_path)
        #在dir_path下创建boxes_info.xml文件
        b_doc = minidom.Document()
        root = b_doc.createElement('boxes_info')
        root.setAttribute('数据时间', time)
        b_doc.appendChild(root)
        camera_position = b_doc.createElement('camera_position')
        camera_position.appendChild(b_doc.createTextNode(i))
        boxes = b_doc.createElement('boxes')
        b_hole = b_doc.createElement('hole')
        #这里应该根据hole数量往b_hole里循环添加数据
        b_target = b_doc.createElement('target')
        #这里应该根据target数量往b_target里循环添加数据
        boxes.appendChild(b_hole)
        boxes.appendChild(b_target)
        imgs = b_doc.createElement('imgs')
        i_hole = b_doc.createElement('hole')
        # 这里应该根据hole数量往b_hole里循环添加数据
        i_target = b_doc.createElement('target')
        # 这里应该根据target数量往b_target里循环添加数据
        imgs.appendChild(i_hole)
        imgs.appendChild(i_target)
        root.appendChild(camera_position)
        root.appendChild(boxes)
        root.appendChild(imgs)
        fp = open(dir_path + '/boxes_info.xml', 'w')
        b_doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        os.mkdir(dir_path+"/boxes_img")

    return str(1)



@testMeasure.route('/pageUp/',methods=['POST','GET'])
def pageUp():
    id = 0
    dirList = []
    time = request.args.get('mydata')
    left_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/A_left')
    #right_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure','static/res_pictures/')
    dirList_f = os.listdir(left_A_path)
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
    left_A_path = '/static/res_pictures/A_left/'+time+'_left.jpg'
    right_A_path = '/static/res_pictures/A_right/'+time+'_right.jpg'
    left_B_path = '/static/res_pictures/B_left/' + time + '_left.jpg'
    right_B_path = '/static/res_pictures/B_right/' + time + '_right.jpg'
    type_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/'+time+'/points_info.xml')
    doc = minidom.parse(type_path)
    tube_name = doc.documentElement.getElementsByTagName("type")[0].childNodes[0].data
    #print(left_path)
    #print(right_path)
    return {"left_A_path":left_A_path,"right_A_path":right_A_path,"left_B_path":left_B_path,"right_B_path":right_B_path,"time":time,"id":id,"num":num,"tube_name":tube_name}




@testMeasure.route('/pageDown/',methods=['POST','GET'])
def pageDown():
    id = 0
    dirList = []
    time = request.args.get('mydata')
    left_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/A_left')
    dirList_f = os.listdir(left_A_path)
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
                i+= 1
                if i >num-1:
                    cur_img = dirList[0]
                    id = 1
                    break
                else:
                    cur_img = dirList[i]
                    id = i+1
                    break
    time = cur_img[:-9]
    left_A_path = '/static/res_pictures/A_left/'+time+'_left.jpg'
    right_A_path = '/static/res_pictures/A_right/'+time+'_right.jpg'
    left_B_path = '/static/res_pictures/B_left/' + time + '_left.jpg'
    right_B_path = '/static/res_pictures/B_right/' + time + '_right.jpg'
    type_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/'+time+'/points_info.xml')
    doc = minidom.parse(type_path)
    tube_name = doc.documentElement.getElementsByTagName("type")[0].childNodes[0].data

    #print(left_path)
    #print(right_path)
    return {"left_A_path":left_A_path,"right_A_path":right_A_path,"left_B_path":left_B_path,"right_B_path":right_B_path,"time":time,"id":id,"num":num,"tube_name":tube_name}


@testMeasure.route('/deleteThis/',methods=['POST','GET'])
def deletThis():
    dirList = []
    time = request.args.get('mydata')
    left_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/A_left/')
    right_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/A_right/')
    left_B_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/B_left/')
    right_B_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/B_right/')
    type_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/')
    command1 = "rm -f "+left_A_path+time+"_left.jpg"
    command2 = "rm -f "+right_A_path+time+"_right.jpg"
    command3 = "rm -f "+left_B_path+ time+"_left.jpg"
    command4 = "rm -f "+right_B_path+time+"_right.jpg"
    command5 = "rm -r "+type_path+time
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    os.system(command5)
    dirList_f = os.listdir(left_A_path)
    for file in dirList_f:
        if os.path.splitext(file)[1] == '.jpg':
            dirList.append(file)
    num = len(dirList)
    # print(dirList)
    cur_img = dirList[-1]
    id = num
    time = cur_img[:-9]
    left_A_path = '/static/res_pictures/A_left/' + time + '_left.jpg'
    right_A_path = '/static/res_pictures/A_right/' + time + '_right.jpg'
    left_B_path = '/static/res_pictures/B_left/' + time + '_left.jpg'
    right_B_path = '/static/res_pictures/B_right/' + time + '_right.jpg'
    type_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure','static/res_pictures/type/' + time + '_type.xml')
    typefs = cv2.FileStorage(type_path, cv2.FileStorage_READ)
    tube_name = typefs.getNode("tube_name").string()
    typefs.release()
    return {"left_A_path":left_A_path,"right_A_path":right_A_path,"left_B_path":left_B_path,"right_B_path":right_B_path,"time":time,"id":id,"num":num,"tube_name":tube_name}


@testMeasure.route('/deleteAll/',methods=['POST','GET'])
def deletAll():
    time = request.args.get('mydata')
    left_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/A_left/')
    right_A_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/A_right/')
    left_B_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/B_left/')
    right_B_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/B_right/')
    type_path = os.path.dirname(os.path.realpath(__file__)).replace('testMeasure', 'static/res_pictures/result/')
    command1 = "rm -f "+left_A_path+"*"
    command2 = "rm -f "+right_A_path+"*"
    command3 = "rm -f "+left_B_path+"*"
    command4 = "rm -f "+right_B_path+"*"
    command5 = "rm -rf "+type_path+"*"
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    os.system(command5)
    return str(1)


@testMeasure.route('/updateList2/',methods=['POST','GET'])
def  updateList2():
    #这里需要加检索筒端类型文件，把检索结果返回到字典里
    result = dict()
    type_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure","static/res_pictures/result")
    type_list = os.listdir(type_path)
    for i in range(1,len(type_list)+1):
        result[i] = type_list[i-1]
    return result


@testMeasure.route('/selectLog/',methods=['POST','GET'])
def  selectLog():
    type = request.args.get('mydata')


    print(type)
    return  str(1)


@testMeasure.route('/targetDetection/',methods=['POST','GET'])
def  targetDetection():
    flag = request.args.get('flag')
    #两步：清空demo中的剩余图片，将目标图片转移到demo
    #运行util
    to_img_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure","tf-faster-rcnn/data/demo/")
    from_img_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure","static/res_pictures/")
    A_left_from_img_path = from_img_path + "A_left/" + flag + "_left.jpg"
    A_right_from_img_path = from_img_path + "A_right/" + flag + "_right.jpg"
    B_left_from_img_path = from_img_path + "B_left/" + flag + "_left.jpg"
    B_right_from_img_path = from_img_path + "B_right/" + flag + "_right.jpg"
    A_left_to_img_path = to_img_path+"A_left.jpg"
    A_right_to_img_path = to_img_path+"A_right.jpg"
    B_left_to_img_path = to_img_path+"B_left.jpg"
    B_right_to_img_path = to_img_path+"B_right.jpg"
    command1 = "rm -f "+to_img_path+"*.jpg"
    command2 = "cp "+A_left_from_img_path+" "+A_left_to_img_path
    command3 = "cp "+A_right_from_img_path+" "+A_right_to_img_path
    command4 = "cp "+B_left_from_img_path+" "+B_left_to_img_path
    command5 = "cp "+B_right_from_img_path+" "+B_right_to_img_path
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    os.system(command5)
    check = get_img_boxes(flag)
    print(flag)
    return  {"key":check}


@testMeasure.route('/centreSight/',methods=['POST','GET'])
def centreSight():
    result = [0,0,0,0,0,0,0,0]
    flag = request.args.get('flag')
    from_path = os.path.dirname(os.path.realpath(__file__)).replace("testMeasure","static/res_pictures/result/")+flag
    # print(boxes_img_path)
    # print("\n")
    im_names = ["A_left", "A_right", "B_left", "B_right"]
    res_score_list = list()
    for i in range(4):
        hole_list = list()
        target_list = list()
        score_list = list()
        boxes_img_dir_path = from_path+"/"+im_names[i]+"/boxes_img"
        save_path = from_path + "/" + im_names[i] + "/boxes_info.xml"
        dirList_f = os.listdir(boxes_img_dir_path)
        for file in dirList_f:
            if os.path.splitext(file)[1] == '.jpg' and (file[:4] == "hole" or file[:4] == "targ"):
                if file[0]=='h':
                    hole_list.append(file)
                elif file[0]=='t':
                    target_list.append(file)
        dom = minidom.parse(save_path)
        root = dom.documentElement
        imgsNode = root.getElementsByTagName('imgs')[0]
        imgsNode.removeChild(root.getElementsByTagName('hole')[1])
        imgsNode.removeChild(root.getElementsByTagName('target')[1])
        imgsNode.appendChild(dom.createElement('hole'))
        imgsNode.appendChild(dom.createElement('target'))
        itemlist_h = root.getElementsByTagName('hole')
        itemlist_t = root.getElementsByTagName('target')
        item_h = itemlist_h[1]
        item_t = itemlist_t[1]
        result[i*2] = numsOf_holes = len(hole_list)
        result[i*2+1] = numsOf_targets = len(target_list)  ## 基准板上三个靶标
        for j in range(numsOf_holes):
            #遍历所有的孔
            imgPath = boxes_img_dir_path+"/hole"+str(j)+".jpg"
            # print(imgPath)
            res = img_process(imgPath,class_of_img="hole")
            hole_img = dom.createElement('hole_img' + str(j))
            x = dom.createElement('x') ##"x"代表横坐标
            x.appendChild(dom.createTextNode(str(res[0])))
            hole_img.appendChild(x)
            y = dom.createElement('y') ##"y"代表纵坐标
            y.appendChild(dom.createTextNode(str(res[1])))
            hole_img.appendChild(y)
            radius = dom.createElement('r') ##"r"代表半径
            radius.appendChild(dom.createTextNode(str(res[2])))
            hole_img.appendChild(radius)
            score = dom.createElement('s') ##"s"代表score，即圆拟合程度
            score.appendChild(dom.createTextNode(str(res[3])))
            hole_img.appendChild(score)
            item_h.appendChild(hole_img)
            score_list.append(res[3])
        for j in range(numsOf_targets):
            #遍历所有的靶标
            imgPath = boxes_img_dir_path+"/target"+str(j)+".jpg"
            #print(imgPath)
            res = img_process(imgPath,class_of_img="target")
            target_img = dom.createElement('target_img' + str(j))
            x = dom.createElement('x')  ##"x"代表横坐标
            x.appendChild(dom.createTextNode(str(res[0])))
            target_img.appendChild(x)
            y = dom.createElement('y')  ##"y"代表纵坐标
            y.appendChild(dom.createTextNode(str(res[1])))
            target_img.appendChild(y)
            radius = dom.createElement('r')  ##"r"代表半径
            radius.appendChild(dom.createTextNode(str(res[2])))
            target_img.appendChild(radius)
            score = dom.createElement('s')  ##"s"代表score，即圆拟合程度
            score.appendChild(dom.createTextNode(str(res[3])))
            target_img.appendChild(score)
            item_t.appendChild(target_img)
            score_list.append(res[3])
        with open(save_path,'w') as fp:
            dom.writexml(fp)
        res_score_list.append(score_list)

    return {"ALh":result[0],"ALt":result[1],"ARh":result[2],"ARt":result[3],"BLh":result[4],"BLt":result[5],"BRh":result[6],"BRt":result[7],"A_left":res_score_list[0],"A_right":res_score_list[1],"B_left":res_score_list[2],"B_right":res_score_list[3]}






