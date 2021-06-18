#-*- coding:utf-8 -*-
import time
import datetime
import  cv2
import  glob
import  numpy  as np
import  os
import  json
from xml.dom import minidom

#  这个函数是要使用的
def sig_calibration(flag,camera_):
    result = list()
    source_dir_path = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/checkboard_img_dir/"+flag+"_"+camera_+"Camera_img_dir/")
    starttime = time.time()
    cbrow = 9
    cbcol = 13    #纵横角点
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)  ##迭代50次或者精度达到0.001
    objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)  ##mgrid处理后尺寸2x9x13（二次元组x9x13），转置后为二次元组x13x9,整体相当于13x9x2顺序数填入objp
    ##print(objp)
    objpoints = []  # 用来存放三维点
    imgpoints = []  # 用来存放图像平面中的二维点
    images = os.listdir(source_dir_path)  # 获取目标目录下所有的图像文件路径
    corner_image_name = 1  ##角点测试图片编号

    print("----------开始进行"+camera_+"相机单目标定-------------")
    ##corner_img_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration', 'corner_img_dir/')  ##已经保存好的图片路径,用来存储角点绘画测试
    for fname in  images:  ##fname是每个图片路径
        if  fname.endswith('.jpg')or fname.endswith('.png'):

            img = cv2.imread(os.path.join(source_dir_path,fname))

            print('********************************************************************',os.path.join(source_dir_path, fname))
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            ret,corners = cv2.findChessboardCorners(gray,(cbrow,cbcol),None)
            if ret:
                ##print('=====================================================================',os.path.join(source_dir_path,fname))
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  ##亚像素定为11x11
                objpoints.append(objp)
                imgpoints.append(corners2)

                img= cv2.drawChessboardCorners(gray,(cbcol,cbrow),corners2,ret)
                corner_img_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','corner_img_dir/')
                corner_img_dir_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','corner_img_dir/'+str(corner_image_name)+'test'+fname+'.jpg')
                cv2.imwrite(corner_img_dir_path,img)
                corner_image_name = corner_image_name+1

    print(len(objpoints))
    print(len(imgpoints))

    new_ojbs = objpoints

    for i,obj in enumerate(objpoints):
        new_ojbs[i] = objpoints[i]*40    ##这里改尺寸



    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(new_ojbs, imgpoints, gray.shape[::-1], None, None)
    h, w = int(3648), int(5472)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  ##产生复原图片
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]  # todo 畸变矫正显示  保存畸变矫正后的图片
    compute_distance_path = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","compute_distance/calibresult.png")
    # todo 畸变矫正显示  保存畸变矫正后的图片
    cv2.imwrite(compute_distance_path, dst)  ##路径错误
    """
    以下是为了评估标定效果
    """
    tot_error = 0
    for num in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[num], rvecs[num], tvecs[num], mtx, dist)
        error = cv2.norm(imgpoints[num], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  ##评估公式,使用欧几里的距离进行评估
        tot_error += error  ##评估积累

    ##if 'leftCamera' in source_dir_path:
        ##camera_file_name = find_camera_calibration_file(source_dir_path) + 'leftCamera'
        ##print(camera_file_name)

    ##elif 'rightCamera' in source_dir_path:
        ##camera_file_name = find_camera_calibration_file(source_dir_path) + 'rightCamera'  ##后续函数
        ##print(camera_file_name)  ##找到左右相机保存路径


    final_object = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'tot_error': tot_error,
        'compute_distance_path': compute_distance_path,
        'corner_img_dir': corner_img_dir
    }

    # 存储单相机标定文件
    print('------------------------------------')
    print('ret', type(ret),
          'rvecs', type(rvecs),
          'tvecs', type(tvecs),
          'tot_error', type(tot_error))
    print('------------------------------------')



    calibration_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/' +flag+'_'+ camera_ + '_' + 'single_calibration.xml')  ##单目标定结果xml文件存放位置
    #calibration_path = '/home/monkiki/PycharmProjects/try/'+camera_ + '_' + 'single_calibration.xml'
    single_calibration_fs = cv2.FileStorage(calibration_path, cv2.FileStorage_WRITE)  ##记住这里要用绝对路径！！！！
    single_calibration_fs.write('ret', ret)
    single_calibration_fs.write('mtx', mtx)
    single_calibration_fs.write('dist', dist)  ##写入固定格式ret、mtx、dist
    """ 存放旋转矩阵 """
    img_num = 1
    for rotation_item in rvecs:
        single_calibration_fs.write("rvecs_" + str(img_num), rotation_item)
        img_num = img_num + 1

    """ 存放平移矩阵 """
    img_num = 1
    for translation_item in tvecs:
        single_calibration_fs.write("tvecs_" + str(img_num), translation_item)
        img_num = img_num + 1

    single_calibration_fs.write('tot_error', tot_error)
    single_calibration_fs.write('compute_distance_path', compute_distance_path)
    single_calibration_fs.write('corner_img_dir', corner_img_dir)

    #  todo 新加入畸变

    single_calibration_fs.release()

    # todo  读取对应的值

    single_calibration_fs_test = cv2.FileStorage(calibration_path, cv2.FileStorage_READ)
    endtime = time.time()
    times = endtime - starttime
    result.append(times)
    print(camera_+'标定完成！总用时'+str(times)+'秒')
    print("===========================================")
    print('ret:', single_calibration_fs_test.getNode('ret').real())  # 正确
    matrix = single_calibration_fs_test.getNode('mtx').mat()
    # result.append([matrix[0, 0], matrix[0, 1], matrix[0, 2]])
    # result.append([matrix[1, 0], matrix[1, 1], matrix[1, 2]])
    # result.append([matrix[2, 0], matrix[2, 1], matrix[2, 2]])
    print(camera_+'相机内参:', matrix)  # 正确  相机内参
    distort = single_calibration_fs_test.getNode('dist').mat()
    # result.append([])
    print(camera_+'相机畸变:', distort)  # 正确  畸变系数
    print(camera_+'旋转矩阵1:', single_calibration_fs_test.getNode('rvecs_1').mat())
    print(camera_+'平移矩阵1:', single_calibration_fs_test.getNode('tvecs_1').mat())
    t_error = single_calibration_fs_test.getNode('tot_error').real()
    print(camera_+'标定效果评估:', t_error)
    result.append(t_error)
    print('compute_distance_path:', single_calibration_fs_test.getNode('compute_distance_path').string())
    print('corner_img_dir:', single_calibration_fs_test.getNode('corner_img_dir').string())



    print(result)
    return result

"""
camera inner parameters
rotate  matrix 
transfer vector 
stereo calibration processing 

"""


def stereo_Calibration(flag):
    result = list()
    starttime = time.time()
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    # Prepare object points
    # todo  修改点的数量
    cbrow = 9
    cbcol = 13  # 纵横角点
    objp = np.zeros((cbcol * cbrow, 3), np.float32)  ##
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Arrays to  store object points and image points from all images
    objpoints_l = []
    objpoints_r = []
    imgpoints_l = []
    imgpoints_r = []

    # Start calibration from the camera
    left_camera_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/')
    leftCamera = left_camera_path+flag+"_leftCamera_img_dir/"
    #left_camera_path = "/home/monkiki/PycharmProjects/stereo_vision/stereo_vision/checkboard_img_dir/leftCamera_img_dir/"
    right_camera_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/')
    rightCamera = right_camera_path+flag+"_rightCamera_img_dir/"
    #right_camera_path = "/home/monkiki/PycharmProjects/stereo_vision/stereo_vision/checkboard_img_dir/rightCamera_img_dir/"
    left_pictures = os.listdir(leftCamera)
    left_pictures.sort()

    right_pictures = os.listdir(rightCamera)
    right_pictures.sort()

    for fname_l, fname_r in zip(left_pictures, right_pictures):
        print(fname_l, fname_r)
        if fname_l.endswith('.jpg') and fname_r.endswith('.jpg'):
            left_path = os.path.join(leftCamera, fname_l)
            print(left_path)
            img_l = cv2.imread(left_path)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

            right_path = os.path.join(rightCamera, fname_r)
            print(right_path)
            img_r = cv2.imread(right_path)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (cbcol, cbrow), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (cbcol, cbrow), None)

            # ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8, 8), None)
            # ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8, 8), None)

            if ret_l and ret_r:
                objpoints_l.append(objp)
                objpoints_r.append(objp)

                corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1),
                                              criteria)
                corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                              criteria)
                # imgpoints_l.append(corners2_l)
                # imgpoints_r.append(corners2_r)

                if (corners2_l[0][0][0] > corners2_l[1][0][0]) and (corners2_r[0][0][0] > corners2_r[1][0][0]):

                    print('corners2_r', corners2_r[0][0][0], corners2_r[1][0][0])

                elif (corners2_l[0][0][0] < corners2_l[1][0][0]):
                    corners2_l = corners2_l[::-1]
                    if (corners2_r[0][0][0] < corners2_r[1][0][0]):
                        corners2_r = corners2_r[::-1]

                elif (corners2_r[0][0][0] < corners2_r[1][0][0]):
                    corners2_r = corners2_r[::-1]

                imgpoints_l.append(corners2_l)
                imgpoints_r.append(corners2_r)

    new_objsl = objpoints_l
    new_objsr = objpoints_r
    for i, obj in enumerate(objpoints_l):
        new_objsl[i] = new_objsl[i] * 40    ##这里调节棋盘格的尺寸
    for i, obj in enumerate(objpoints_r):
        new_objsr[i] = new_objsr[i] * 40    ##这里调节棋盘格的尺寸

    ret, mtx_l, dist_l, rvecs, tvecs = cv2.calibrateCamera(new_objsl, imgpoints_l,
                                                           gray_l.shape[::-1], None,
                                                           None)
    h, w = int(3648), int(5472)

    newcameramtx_l, roi = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, (w, h), 1,
                                                        (w, h))
    ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(new_objsr,
                                                           imgpoints_r,
                                                           gray_r.shape[::-1],
                                                           None, None)

    newcameramtx_r, roi = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w, h),
                                                        1, (w, h))

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(new_objsl,
                                                                                                     imgpoints_l,
                                                                                                     imgpoints_r, mtx_l,
                                                                                                     dist_l, mtx_r,
                                                                                                     dist_r,
                                                                                                     gray_l.shape[::-1])

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                                                                      distCoeffs2, gray_l.shape[::-1], R, T)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                       distCoeffs1, R1, P1,
                                                       gray_l.shape[::-1],
                                                       cv2.INTER_NEAREST)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                         distCoeffs2, R2,
                                                         P2, gray_l.shape[::-1],
                                                         cv2.INTER_NEAREST)

    calibration_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
                                                                           'static/checkboard_img_dir/'+flag+'_'+'stereo_calibration.xml')

    stereo_calibration_fs = cv2.FileStorage(
        calibration_path,
        cv2.FileStorage_WRITE)

    # stereo_calibration_fs.write('mtx_stereo', MLS)
    # stereo_calibration_fs.write('dts_stereo', dLS)
    stereo_calibration_fs.write('rective_stereo', Q)
    stereo_calibration_fs.write('fundamental_matrix', F)
    stereo_calibration_fs.write('left_stereo_map0', left_map1)
    stereo_calibration_fs.write('left_stereo_map1', left_map2)
    stereo_calibration_fs.write('right_stereo_map0', right_map1)
    stereo_calibration_fs.write('right_stereo_map1', right_map2)
    stereo_calibration_fs.write('R_matrix', R)
    stereo_calibration_fs.write('T_matrix', T)
    stereo_calibration_fs.write('P1', P1)
    stereo_calibration_fs.write('P2', P2)
    print('R', R)
    print('T', T)
    stereo_calibration_fs.release()
    endtime = time.time()
    st = datetime.datetime.fromtimestamp(endtime).strftime('%Y' + '-' + '%m' + '-' + '%d' + '-' + '%H' + ':' + '%M' + ':' + '%S')
    global_path = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/global_info.xml")
    if flag=="A":
        dom = minidom.parse(global_path)
        root = dom.documentElement
        calibration_time = root.getElementsByTagName("A_calibration_time")[0]
        calibration_time.removeChild(root.getElementsByTagName("time")[0])
        time1 = dom.createElement("time")
        calibration_time.appendChild(time1)
        time1.appendChild(dom.createTextNode(st))
        with open(global_path, 'w') as fp:
            dom.writexml(fp)
    elif flag=="B":
        dom = minidom.parse(global_path)
        root = dom.documentElement
        calibration_time = root.getElementsByTagName("B_calibration_time")[0]
        calibration_time.removeChild(root.getElementsByTagName("time")[1])
        time1 = dom.createElement("time")
        calibration_time.appendChild(time1)
        time1.appendChild(dom.createTextNode(st))
        with open(global_path, 'w') as fp:
            dom.writexml(fp)
    times = endtime - starttime
    print("双目标定完成！用时" + str(times) + "秒")
    result.append(times)


    print(result)
    return result


def stereo_Calibration_for_use(left_name, right_name):

    calibration_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration',
                                                                           'checkboard_img_dir/' + 'stereo_calibration.xml')
    stereo_calibration_fs = cv2.FileStorage(calibration_path,cv2.FileStorage_READ)
    '''
    Left_Stereo_Map = np.zeros(2)
    Right_Stereo_Map = np.zeros(2)
    Left_Stereo_Map = np.concatenate(Left_Stereo_Map, axis = 0)
    Right_Stereo_Map = np.concatenate(Right_Stereo_Map, axis = 0)
    '''

    base_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/zj_pictures/stereo_calibration/')


    # list(range(1000))
    Left_Stereo_Map = ['', '']
    Right_Stereo_Map = ['', '']
    Left_Stereo_Map[0] = stereo_calibration_fs.getNode('left_stereo_map0').mat()
    Left_Stereo_Map[1] = stereo_calibration_fs.getNode('left_stereo_map1').mat()
    Right_Stereo_Map[0] = stereo_calibration_fs.getNode('right_stereo_map0').mat()
    Right_Stereo_Map[1] = stereo_calibration_fs.getNode('right_stereo_map1').mat()
    Q                   = stereo_calibration_fs.getNode('rective_stereo').mat()
    stereo_calibration_fs.release()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #img_left = cv2.imread(left_name)
    #gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    #ret, corners = cv2.findChessboardCorners(gray, (13, 9), None)
    #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #img = cv2.drawChessboardCorners(gray, (13, 9), corners2, ret)
    #cv2.imwrite(os.path.join(base_path, "corners_left.jpg"), img)


    # todo 画棋盘格的程序
    #img_right = cv2.imread(right_name)
    #gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    #ret, corners = cv2.findChessboardCorners(gray, (13, 9), None)
    #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #img = cv2.drawChessboardCorners(gray, (13, 9), corners2, ret)
    #cv2.imwrite(os.path.join(base_path, "corners_right.jpg"), img)
    # Q矩阵为重投影矩阵
    print(Q)
    img_right = cv2.imread(right_name)
    img_left = cv2.imread(left_name)
    img = cv2.remap(img_left, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
    img2 = cv2.remap(img_right, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)

    # 灰度图准备
    imgL = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 根据Block Maching方法生成差异图
    stereo = cv2.StereoBM_create(numDisparities=144, blockSize=5)
    disparity = stereo.compute(imgL, imgR)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)



    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 32., Q)



    min_disp= 2
    num_disp = 130 - min_disp

    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp


    cv2.imwrite(os.path.join(base_path, 'stereo_remap_left.jpg'), img)
    cv2.imwrite(os.path.join(base_path, "stereo_remap_right.jpg"), img2)
    cv2.imwrite(os.path.join(base_path, "disp.jpg"), disparity)
    cv2.imwrite(os.path.join(base_path, "disparity.jpg"), threeD)
    cv2.imwrite(os.path.join(base_path, "depth.jpg"), disp)

    cv2.namedWindow('',0)
    cv2.resizeWindow('',(800,600))
    cv2.imshow('',disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




"""
find the camerafile path in the given path
"""
def  find_camera_calibration_file(path):

     find_path = str(path).split('/',)
     find_path = find_path[0]+'/'+find_path[1]+'/'+find_path[2]+'/'+find_path[3]+'/'+find_path[4]+'/'+find_path[5]+'/'

     return find_path

"""
check up  txt file  if not exist then create it  
if exist  return true 
"""
def  check_up_txt_file(path):

     files = {
         'ret'             :str(path +'ret.txt'),
         'mtx'             :str(path + 'mtx.txt'),
         'dist'            :str(path + 'dist.txt'),
         'rvecs'           :str(path + 'rvecs.txt'),
         'tvecs'           :str(path + 'tvecs.txt'),
         'tot_error'            :str(path + 'tot_error.txt'),
         'compute_distance_path':str(path + 'compute_distance_path.txt'),
         'corner_img_dir'       :str(path + 'corner_img_dir.txt'),

     }
     files_name =  ['ret_path','mtx_path','dist_path','rvecs_path','tvecs_path','tot_error','compute_distance_path']
     return files.values()

def   save_object(object, file_path):

        if type(object)==type('123')      :
           with open(file_path, 'w')  as f:
               f.write(object)
        elif type(object)== type(float(123))   :
           with open(file_path, 'w')  as f:
               f.write(str(object))
        elif type(object)== type([1,2,3,4])   :
           with open(file_path, 'w')  as f:
                for line in  object:
                    f.write(str(line)+'\n')
        else:
            np.save(file_path,object)


# 由两幅图像生成视差图
def   creat_disp(left_path,right_path):

       left = cv2.imread(left_path,0)
       right = cv2.imread(right_path,0)
       stereo = cv2.StereoBM_create(numDisparities=144, blockSize=5)
       disparity = stereo.compute(left,right)

       return  disparity


def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)

    AA = A - np.tile(mu_A, (N, 1))
    BB = B - np.tile(mu_B, (N, 1))
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * mu_A.T + mu_B.T

    return R, t

def LTOrd2AOrd(AP1,AP2,AP3,BP1,BP2,BP3):
    B = np.mat([[0,0,0],[260.004,0,0],[0,130.007,0]])  ##数据根据基准板尺寸修改,需要考虑实际加工公差
    A = np.mat([[AP1[0,0],AP1[0,1],AP1[0,2]],[AP2[0,0],AP2[0,1],AP2[0,2]],[AP3[0,0],AP3[0,1],AP3[0,2]]])

    R_T2A,T_T2A = rigid_transform_3D(A,B)
    n = len(A)
    # B2 = (R_T2A * A.T) + np.tile(T_T2A, (1, n))
    # B2 = B2.T
    # print("B2\n")
    # print(B2)
    C = np.mat([[BP1[0,0],BP1[0,1],BP1[0,2]],[BP2[0,0],BP2[0,1],BP2[0,2]],[BP3[0,0],BP3[0,1],BP3[0,2]]])
    C2 = (R_T2A * C.T) + np.tile(T_T2A, (1, n))
    C2 = C2.T
    # print("C2\n")
    # print(C2)
    APO = B[0]
    APH = B[1]
    APW = B[2]
    BPO = C2[0]
    BPH = C2[1]
    BPW = C2[2]
    return APO,APH,APW,BPO,BPH,BPW,R_T2A,T_T2A


def LTside2VSide(APO,APH,APW,BPO,BPH,BPW):
    A_x_axial = APH - APO
    A_y_axial = APW - APO
    B_x_axial = BPH - BPO
    B_y_axial = BPW - BPO
    mT_AS2S = np.cross(A_y_axial,A_x_axial)
    mT_BS2S = np.cross(B_y_axial,B_x_axial)
    m = 50.0   ##定长m为实际测量值
    T_AS2S = mT_AS2S/np.linalg.norm(mT_AS2S)*m
    T_BS2S = mT_BS2S/np.linalg.norm(mT_BS2S)*m
    print("T_AS2S:===========")
    print(T_AS2S)
    print("T_BS2S:===========")
    print(T_BS2S)
    APOs = APO + T_AS2S
    APHs = APH + T_AS2S
    APWs = APW + T_AS2S
    BPOs = BPO + T_BS2S
    BPHs = BPH + T_BS2S
    BPWs = BPW + T_BS2S
    return APOs,APHs,APWs,BPOs,BPHs,BPWs,T_AS2S,T_BS2S




























