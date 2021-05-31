import numpy as np
import os
import time
import math
from xml.dom import minidom

from stereo_vision.cameraCalibration.utils import rigid_transform_3D ##这里要改回来

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
    if rho[2] < 0:
        return 360 - theta
    else:
        return theta


def tubePoseCalculate(filePath,dataPath):
    """
    :param filePath:
    :param dataPath:
    :return:
    A：
        拿到的是计算值路径与先验数据文件路径
        1、解析相应文件
            求两方数据安装孔的数量及各自p_index，先验数据象限孔的数量
            找出外标的坐标
        2、判断先验数据种类，根据有无象限孔分为两类
            ①有象限孔（外标装在象限孔上）
            安装孔数是否相等，否就重拍
            求出计算值这边的安装孔质心坐标
            求外标向量
            两方偏角与p_index一一对应，以外标向量为基准顺时针旋转
            分别根据偏角对p_index asc，这时得到的分别是计算值index的排列和先验数据index的排列，数量应该相等
            根据相应的排列按照顺序生成两组点云mat(注意先验此时是二维点，需要先转换成三维)
            求转换关系
            略
            ②无象限孔（外标装在右边第一个安装孔上）
            安装孔数是否差一，否就重拍（先验比计算值多1）
            加入外标求出计算值这边的安装孔质心坐标
            求外标向量
            两方偏角与p_index一一对应，以外标向量为基准顺时针旋转
            分别根据偏角对p_index asc，去掉先验值这边的0偏角，即对应的外标本身向量，此时两方数量应该相等
            根据相应的排列按照顺序生成两组点云mat（注意先验数据此时是二维点，需要先转换成三维）
            求转换关系
            略
    B：
        略
    """
    globalPath = os.path.dirname(os.path.realpath(__file__)).replace("poseCalculate","static/global_info.xml")
    p_doc = minidom.parse(filePath)
    d_doc = minidom.parse(dataPath)
    g_doc = minidom.parse(globalPath)
    p_root = p_doc.documentElement
    d_root = d_doc.documentElement
    g_root = g_doc.documentElement
    ##注意这块只处理A端:
    ##处理理论值
    Amouting = d_root.getElementsByTagName('Amouting')[0]
    Atapped = d_root.getElementsByTagName('Atapped')[0]
    Aquadrant = d_root.getElementsByTagName('Aquadrant')[0]
    numsOfAmouting = len(Amouting.childNodes)
    numsOfAtapped = len(Atapped.childNodes)
    numsOfAquadrant = len(Aquadrant.childNodes)
    # print("numsOfAmouting:"+str(numsOfAmouting))
    # print("numsOfAtapped:"+str(numsOfAtapped))
    # print("numsOfAquadrant:"+str(numsOfAquadrant))
    ##处理计算值
    Ahole = p_root.getElementsByTagName("threeD")[0]
    Atarget = p_root.getElementsByTagName("threeD")[1]
    numsOfAhole = len(Ahole.childNodes)
    numsOfAtarget = len(Atarget.childNodes)## 此处默认检测到的靶标数量为4(3个板上1个筒上)
    # print("numsOfAhole:"+str(numsOfAhole))
    data_list = list()
    data_index_list = list()
    for i in range(numsOfAmouting):
        x = Amouting.getElementsByTagName('p' + str(i))[0].childNodes[0].childNodes[0].data
        y = Amouting.getElementsByTagName('p' + str(i))[0].childNodes[1].childNodes[0].data
        data_list.append([float(x),float(y)])
        data_index_list.append(i)

    hole_list = list()
    hole_index_list = list()
    hole_r_list = list()
    for i in range(numsOfAhole):
        x = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[0].childNodes[0].data
        y = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[1].childNodes[0].data
        z = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[2].childNodes[0].data
        r = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[3].childNodes[0].data
        hole_list.append([float(x),float(y),float(z)])
        hole_index_list.append(i)
        hole_r_list.append(float(r))
    # print(data_list)
    # print(data_index_list)
    # print(hole_list)
    # print(hole_index_list)
    # print(hole_r_list)
    ##已验证
    target_list = list()
    target_index_list = list()
    ##生成靶标列表的同时找出离群靶标即外靶标
    record_i = 0
    max_y = float('-inf')
    for i in range(numsOfAtarget):
        x = Atarget.getElementsByTagName('point' + str(i))[0].childNodes[0].childNodes[0].data
        y = float(Atarget.getElementsByTagName('point' + str(i))[0].childNodes[1].childNodes[0].data)
        z = Atarget.getElementsByTagName('point' + str(i))[0].childNodes[2].childNodes[0].data
        target_list.append([float(x), float(y), float(z)])
        target_index_list.append(i)
        if y>max_y:
            max_y = y
            record_i = i
    # print(target_list)
    # print(target_index_list)
    distence = d = 0
    Outlier_i = 0
    for i in range(numsOfAtarget):
        d = (target_list[i][0]-target_list[record_i][0])**2+(target_list[i][1]-target_list[record_i][1])**2+(target_list[i][2]-target_list[record_i][2])**2
        if d>distence:
            Outlier_i = i
            distence = d
    # print("离群点与板上最下靶标距离："+str(math.sqrt(distence)))
    # print("外标在index中id："+str(Outlier_i))
    # print("外标坐标："+str(target_list[Outlier_i]))
    ##已验证
    ##分情况讨论
    if numsOfAquadrant!=0:
        ##有象限孔
        if numsOfAmouting+numsOfAtapped+numsOfAquadrant!=numsOfAhole+1:
            return False
        ##根据孔径估算排名得出安装孔实测列表
        hole_index_list = np.argsort(hole_r_list)[::-1]
        hole_index_list = hole_index_list[:numsOfAmouting]
        p_Amouting = list()
        p_Amouting_mat = np.mat(np.zeros([numsOfAmouting,3]))
        for i in hole_index_list:
            p_Amouting.append(hole_list[i]) ##得出计算值中的安装孔列表
            p_Amouting_mat[i, 0] = hole_list[i][0]
            p_Amouting_mat[i, 1] = hole_list[i][1]
            p_Amouting_mat[i, 2] = hole_list[i][2]
        mu_p_Amouting = np.mean(p_Amouting_mat,axis=0)
        # print("计算值质心坐标："+str(mu_p_Amouting))
        # print("安装孔数量："+str(len(p_Amouting)))
        fir_vector = target_list[Outlier_i]-mu_p_Amouting
        # print("外标向量："+str(fir_vector))
        ##已检验
        ##定义偏角列表
        angle_list = list()
        for i in range(len(p_Amouting)):
            angleOfmouting = GetClockAngle(np.squeeze(np.asarray(p_Amouting[i]-mu_p_Amouting)),np.squeeze(np.asarray(fir_vector)))
            angle_list.append(angleOfmouting)
        # print("安装孔偏角列表"+str(angle_list))
        ##已验证
        p_index_Amouting = np.argsort(angle_list)[::-1] ##因为y轴相反，所以降序代替升序
        aft_p_Amouting_mat = np.mat(np.zeros([len(p_Amouting),3]))
        for i in range(len(p_Amouting)):
            aft_p_Amouting_mat[i, 0] = p_Amouting[p_index_Amouting[i]][0]
            aft_p_Amouting_mat[i, 1] = p_Amouting[p_index_Amouting[i]][1]
            aft_p_Amouting_mat[i, 2] = p_Amouting[p_index_Amouting[i]][2]
        # print(p_index_Amouting)
        # print(aft_p_Amouting_mat)
        ##已验证
        record_target = target_list[record_i]
        target_list.pop(Outlier_i)
        p_width_index = 0
        # print(target_list)
        ##找出每个靶标的身份
        distence = float("-inf")
        p_height_index = 0
        for i in range(len(target_list)):
            d = (target_list[i][0]-record_target[0])**2+(target_list[i][1]-record_target[1])**2+(target_list[i][2]-record_target[2])**2
            if d>distence:
                distence = d
                p_height_index = i
            if d==0:
                p_width_index = i
        p_origin_index = 3-(p_width_index+p_height_index)
        # print("APH"+str(target_list[p_height_index]))
        # print("APW"+str(target_list[p_width_index]))
        # print("APO"+str(target_list[p_origin_index]))
        ##已检验
        APOs = g_root.getElementsByTagName("APOs")[0]
        APHs = g_root.getElementsByTagName("APHs")[0]
        APWs = g_root.getElementsByTagName("APWs")[0]
        APOs_X = float(APOs.getElementsByTagName("X")[0].childNodes[0].data)
        APOs_Y = float(APOs.getElementsByTagName("Y")[0].childNodes[0].data)
        APOs_Z = float(APOs.getElementsByTagName("Z")[0].childNodes[0].data)
        APHs_X = float(APHs.getElementsByTagName("X")[0].childNodes[0].data)
        APHs_Y = float(APHs.getElementsByTagName("Y")[0].childNodes[0].data)
        APHs_Z = float(APHs.getElementsByTagName("Z")[0].childNodes[0].data)
        APWs_X = float(APWs.getElementsByTagName("X")[0].childNodes[0].data)
        APWs_Y = float(APWs.getElementsByTagName("Y")[0].childNodes[0].data)
        APWs_Z = float(APWs.getElementsByTagName("Z")[0].childNodes[0].data)
        Atarget_mat = np.mat([[APOs_X,APOs_Y,APOs_Z],[APHs_X,APHs_Y,APHs_Z],[APWs_X,APWs_Y,APWs_Z]])
        p_Atarget_mat = np.mat([[target_list[p_origin_index][0],target_list[p_origin_index][1],target_list[p_origin_index][2]],[target_list[p_height_index][0],target_list[p_height_index][1],target_list[p_height_index][2]],[target_list[p_width_index][0],target_list[p_width_index][1],target_list[p_width_index][2]]])
        # print(Atarget_mat)
        # print(p_Atarget_mat)
        ##已检验
        # print(aft_p_Amouting_mat)
        R_p2g,T_p2g = rigid_transform_3D(p_Atarget_mat,Atarget_mat)
        n = len(aft_p_Amouting_mat)
        aft_p_Amouting_mat2 = (R_p2g * aft_p_Amouting_mat.T) + np.tile(T_p2g, (1, n))
        aft_p_Amouting_mat2 = aft_p_Amouting_mat2.T
        # print(aft_p_Amouting_mat2)
        ##以上完成了安装孔按角度排序且转换至全局坐标系
        ##还没排序
        d_Amouting = list()
        for i in range(len(data_list)):
            d_Amouting.append([data_list[i][0],data_list[i][1],0])
        # print(d_Amouting)
        centerAndAngle = np.mat(np.zeros([2,3]))
        d_center = d_root.getElementsByTagName("center")[0]
        d_center_X = float(d_center.getElementsByTagName("x")[0].childNodes[0].data)
        d_center_Y = float(d_center.getElementsByTagName("y")[0].childNodes[0].data)
        mu_Amouting = np.mat([d_center_X,d_center_Y,0])
        targetAquadrant = [float(Aquadrant.getElementsByTagName("x")[0].childNodes[0].data),float(Aquadrant.getElementsByTagName("y")[0].childNodes[0].data),0]
        d_fir_vector = targetAquadrant - mu_Amouting
        # print(d_fir_vector)
        ##已检验
        d_angle_list = list()
        for i in range(len(d_Amouting)):
            angleOfmouting = GetClockAngle(np.squeeze(np.asarray(d_Amouting[i]-mu_Amouting)),np.squeeze(np.asarray(d_fir_vector)))
            d_angle_list.append(angleOfmouting)
        # print("安装孔偏角列表"+str(d_angle_list))
        d_index_Amouting = np.argsort(d_angle_list)
        aft_d_Amouting_mat = np.mat(np.zeros([len(d_Amouting), 3]))
        for i in range(len(d_Amouting)):
            aft_d_Amouting_mat[i, 0] = d_Amouting[d_index_Amouting[i]][0]
            aft_d_Amouting_mat[i, 1] = d_Amouting[d_index_Amouting[i]][1]
            aft_d_Amouting_mat[i, 2] = d_Amouting[d_index_Amouting[i]][2]
        # print(d_index_Amouting)
        # print(aft_d_Amouting_mat)
        ##已检验
        ##下一步求d2p转换关系







        centerAndAngle[0, 0] = d_center_X
        centerAndAngle[0, 1] = d_center_Y


    else:
        ##无象限孔
        if numsOfAmouting+numsOfAtapped+numsOfAquadrant!=numsOfAhole+1:
            return False

    return str(1)