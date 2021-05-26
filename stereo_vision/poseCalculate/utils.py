import numpy as np
import os
import time
from xml.dom import minidom

from stereo_vision.cameraCalibration.utils import rigid_transform_3D ##这里要改回来




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
    ##注意这块只处理A端:
    p_doc = minidom.parse(filePath)
    d_doc = minidom.parse(dataPath)
    p_root = p_doc.documentElement
    d_root = d_doc.documentElement
    ##处理理论值
    Amouting = d_root.getElementsByTagName('Amouting')[0]
    Atapped = d_root.getElementsByTagName('Atapped')[0]
    Aquadrant = d_root.getElementsByTagName('Aquadrant')[0]
    numsOfAmouting = len(Amouting.childNodes)
    numsOfAtapped = len(Atapped.childNodes)
    numsOfAquadrant = len(Aquadrant.childNodes)
    print("numsOfAmouting:"+str(numsOfAmouting))
    print("numsOfAtapped:"+str(numsOfAtapped))
    print("numsOfAquadrant:"+str(numsOfAquadrant))
    ##处理计算值
    Ahole = p_root.getElementsByTagName("threeD")[0]
    numsOfAhole = len(Ahole.childNodes)
    print("numsOfAhole:"+str(numsOfAhole))
    data_list = list()
    data_index_list = list()
    for i in range(numsOfAmouting):
        x = Amouting.getElementsByTagName('p' + str(i))[0].childNodes[0].childNodes[0].data
        y = Amouting.getElementsByTagName('p' + str(i))[0].childNodes[1].childNodes[0].data
        data_list.append([float(x),float(y)])
        data_index_list.append(i)

    point_list = list()
    point_index_list = list()
    point_r_list = list()
    for i in range(numsOfAhole):
        x = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[0].childNodes[0].data
        y = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[1].childNodes[0].data
        z = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[2].childNodes[0].data
        r = Ahole.getElementsByTagName('point' + str(i))[0].childNodes[3].childNodes[0].data
        point_list.append([float(x),float(y),float(z)])
        point_index_list.append(i)
        point_r_list.append(float(r))
    print(data_list)
    print(data_index_list)
    print(point_list)
    print(point_index_list)
    print(point_r_list)
    ##分情况讨论






    return str(1)





if __name__=="__main__":
    filePath = "/home/monkiki/PycharmProjects/measurement_project/stereo_vision/static/res_pictures/result/2021-01-16-16:03:26/points_info.xml"
    dataPath = "/home/monkiki/PycharmProjects/measurement_project/stereo_vision/static/priori_data/2021-5-19-22:42:12.xml"
    tubePoseCalculate(filePath, dataPath)