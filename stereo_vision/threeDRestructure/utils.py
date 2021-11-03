import os
import cv2
import numpy as np
import math
from xml.dom import minidom
from stereo_vision.prioriImport.utils import fitCircle


def  point_undistort(point,flag,end='A'):
    if end=='A':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure', 'static/checkboard_img_dir/A_')
    elif end=='B':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure', 'static/checkboard_img_dir/B_')
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
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure','static/checkboard_img_dir/A_')
    elif end=='B':
        root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure','static/checkboard_img_dir/B_')
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



def   find_Fmatrix(epoch_name): #涉及函数1

    calibration_path_A = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure',
                                                                        'static/checkboard_img_dir/' + 'A_stereo_calibration.xml')
    calibration_path_B = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure',
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


# 函数的作用:left_point在right_points里面的最匹配点,设置了阈值，
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

## 类比此函数的功能，写出另外两个，这个不用了
def creat_point_pairs(right_points, left_points, F):
    all_pairs = []
    for left_point in left_points:
        index, right_point = min_distance_pnt(right_points, left_point, F)  ##找出对应当前左点的最佳右点
        # todo 查看right_point数据格式
        print(len(right_points))
        if len(right_points) <= 0:
            break
        right_points.pop(index)
        # pair['left_point'] = left_point
        # pair['right_point'] = right_point
        # print('left_point', left_point)
        # print('right_point',right_point)
        left_x = left_point[0]
        left_y = left_point[1]
        right_x = right_point[0]
        right_y = right_point[1]
        radius = (left_point[2] + right_point[2]) // 2
        score = (left_point[3] + right_point[3]) // 2
        all_pairs.append([[left_x, left_y], [right_x, right_y], radius, score])
    print("=====")
    return all_pairs


def findOHW(p1,p2,p3,p4):
    ##先选出横坐标接近的两个点，最终认为p1是PO、p2是PH、p3是PW、p4是筒上靶标
    ##比较顺序： p1->p2\p3\p4 p2->p3\p4 p3->p4
    dis_array = [0]*6
    dis_array[0] = abs(p1[0] - p2[0])
    dis_array[1] = abs(p1[0] - p3[0])
    dis_array[2] = abs(p1[0] - p4[0])
    dis_array[3] = abs(p2[0] - p3[0])
    dis_array[4] = abs(p2[0] - p4[0])
    dis_array[5] = abs(p3[0] - p4[0])
    minnum = dis_array[0]
    minindex = 0
    for i in range(6):
        if dis_array[i]<=minnum:
            minnum = dis_array[i]
            minindex = i
    if minindex==0:
        ##p1\p2
        if p1[1]<p2[1]:
            _p = p2
            p2 = p3
            p3 = _p
        else:
            _p = p1
            p1 = p2
            p2 = p3
            p3 = _p
    elif minindex==1:
        ##p1\p3
        if p1[1]<p3[1]:
            p1 = p1
        else:
            _p = p1
            p1 = p3
            p3 = _p
    elif minindex==2:
        ##p1\p4
        if p1[1]<p4[1]:
            _p = p4
            p4 = p3
            p3 = _p
        else:
            _p = p1
            p1 = p4
            p4 = p3
            p3 = _p
    elif minindex==3:
        ##p2\p3
        if p2[1]<p3[1]:
            _p = p2
            p2 = p1
            p1 = _p
        else:
            _p = p1
            p1 = p3
            p3 = p2
            p2 = _p
    elif minindex==4:
        ##p2\p4
        if p2[1]<p4[1]:
            _p = p1
            p1 = p2
            p2 = _p
            _p = p4
            p4 = p3
            p3 = _p
        else:
            _p = p1
            p1 = p4
            p4 = _p
            _p = p2
            p2 = p3
            p3 = _p
    else:
        ##p3\p4
        if p3[1]<p4[1]:
            _p = p1
            p1 = p3
            p3 = p4
            p4 = _p
        else:
            _p = p1
            p1 = p4
            p4 = _p
    ##此时p1、p3已经确定
    if abs(p2[0]-p1[0])>abs(p4[0]-p1[0]):
        _p = p2
        p2 = p4
        p4 = _p
    return p1,p2,p3,p4


def creat_target_pairs(right_points,left_points):
    ##left_points和right_points的结构都为[x,y,radius,score]且必然为四个
    ##返回all_pairs列表
    all_pairs = list()
    left_points[0],left_points[1],left_points[2],left_points[3] = findOHW(left_points[0],left_points[1],left_points[2],left_points[3])
    right_points[0],right_points[1],right_points[2],right_points[3] = findOHW(right_points[0],right_points[1],right_points[2],right_points[3])
    for i in range(4):
        all_pairs.append([[left_points[i][0],left_points[i][1]],[right_points[i][0],right_points[i][1]],(left_points[i][2]+right_points[i][2])//2,(left_points[i][3]+right_points[i][3])//2])
    return all_pairs,left_points[3],right_points[3]



def GetClockAngle(v1, v2):  ##求v2到v1顺时针旋转角度
    # print(v1)
    # print(v2)
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    # 叉乘
    rho =  np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
    # print(rho)
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
    # print(theta)
    if rho < 0:
        return 360 - theta
    else:
        return theta


def creat_hole_pairs(right_points,left_points,leftp4,rightp4):
    ##返回all_pairs列表
    all_pairs = list()
    l_x,l_y,r_x,r_y = np.zeros(len(left_points)+1),np.zeros(len(left_points)+1),np.zeros(len(right_points)+1),np.zeros(len(right_points)+1)
    for i in range(len(left_points)):
        l_x[i] = left_points[i][0]
        l_y[i] = left_points[i][1]
        r_x[i] = right_points[i][0]
        r_y[i] = right_points[i][1]
    l_x[len(left_points)] = leftp4[0]
    l_y[len(left_points)] = leftp4[1]
    r_x[len(right_points)] = rightp4[0]
    r_y[len(right_points)] = rightp4[1]
    l_xc, l_yc, l_R, l_residu = fitCircle(l_x, l_y)
    r_xc, r_yc, r_R, r_residu = fitCircle(r_x, r_y)
    left_angle_list = list()
    right_angle_list = list()
    ##定义了夹角列表，先把夹角求出来后得出排序索引
    lfir_vactor = [leftp4[0] - l_xc , leftp4[1]-l_yc]
    rfir_vactor = [rightp4[0] - r_xc , rightp4[1] - r_yc]
    for i in range(len(left_points)):
        left_angle_list.append(GetClockAngle(np.squeeze(np.asarray([left_points[i][0] - l_xc , left_points[i][1] - l_yc])),np.squeeze(np.asarray(lfir_vactor))))
        right_angle_list.append(GetClockAngle(np.squeeze(np.asarray([right_points[i][0] - r_xc, right_points[i][1] - r_yc])),np.squeeze(np.asarray(rfir_vactor))))
    leftsortlist = np.argsort(left_angle_list)
    rightsortlist = np.argsort(right_angle_list)
    for i in range(len(left_points)):
        leftindex = leftsortlist[i]
        rightindex = rightsortlist[i]
        all_pairs.append([[left_points[leftindex][0],left_points[leftindex][1]],[right_points[rightindex][0],right_points[rightindex][1]],(left_points[leftindex][2]+right_points[rightindex][2])//2,(left_points[leftindex][3]+right_points[rightindex][3])//2])
    return all_pairs





# 修改后把匹配好的左右点存入points_info.xml
def save_pairs_file(epoch_name,pairs,end='A',hot="hole"):
    ##主要步骤是先擦除原始数据，再往里面写数据
    root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure', 'static/res_pictures/result/')
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
    root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure', 'static/res_pictures/result/')
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

    ##这里做一下适应性修改，因为标定问题，对极约束并不是很好用，留作选用函数，改用适用于筒端一周的角度排序匹配法
    ##定义两函数：1、creat_target_pairs 2、creat_hole_pairs
    ##首先是利用粗略位置关系的creat_target_pairs，拿到二维图片中插在筒端上的视觉靶标
    ##其次是利用向量关系求解角度并排序的creat_hole_pairs,最终完成跟之前一样格式的匹配结果


    A_targets_pairs,AleftTargetPoint,ArightTargetPoint = creat_target_pairs(AR_targets, AL_targets)
    B_targets_pairs,BleftTargetPoint,BrightTargetPoint = creat_target_pairs(BR_targets, BL_targets)
    ##target匹配已经改完


    A_holes_pairs = creat_hole_pairs(AR_holes, AL_holes, AleftTargetPoint,ArightTargetPoint)  ##涉及函数3
    B_holes_pairs = creat_hole_pairs(BR_holes, BL_holes, BleftTargetPoint,BrightTargetPoint)



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
    result = dict()
    data_root_path = os.path.dirname(os.path.realpath(__file__)).replace('threeDRestructure', 'static/res_pictures/result/')
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
    res = list()
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
        res.append([round(xyz[0][0],3),round(xyz[1][0],3),round(xyz[2][0],3)])
    result["AH"] = res
    ## 计算A端标的三维坐标
    res = list()
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
        res.append([round(xyz[0][0],3),round(xyz[1][0],3),round(xyz[2][0],3)])
    result["AT"] = res
    ## 计算B端孔的三维坐标
    res = list()
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
        res.append([round(xyz[0][0],3),round(xyz[1][0],3),round(xyz[2][0],3)])
    result["BH"] = res
    ## 计算B端标的三维坐标
    res = list()
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
        res.append([round(xyz[0][0],3),round(xyz[1][0],3),round(xyz[2][0],3)])
    result["BT"] = res

    with open(pairs_3D_path,'w') as fp:
        dom.writexml(fp)

    return result