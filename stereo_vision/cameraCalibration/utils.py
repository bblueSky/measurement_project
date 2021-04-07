#-*- coding:utf-8 -*-
import time
import  cv2
import  glob
import  numpy  as np
import  os
import  json


#  这个函数是要使用的
def sig_calibration(flag,camera_):
    source_dir_path = os.path.dirname(os.path.realpath(__file__)).replace("cameraCalibration","static/checkboard_img_dir/"+flag+"_"+camera_+"Camera_img_dir/")
    starttime = time.time()
    cbrow = 9
    cbcol = 13    #纵横角点
    objp = np.zeros((cbrow * cbcol, 3), np.float32)  ##objp尺寸为（角点总数x3）
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
            """角点精确化迭代过程的终止条件"""
            """执行亚像素级角点检测"""
            ##print('=====================================================================',os.path.join(source_dir_path,fname))
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  ##亚像素定为11x11
            objpoints.append(objp)
            imgpoints.append(corners2)
            """在棋盘上绘制角点，只是可视化工具"""
            img= cv2.drawChessboardCorners(gray,(13,9),corners2,ret)
            corner_img_dir = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','corner_img_dir/')
            corner_img_dir_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','corner_img_dir/'+str(corner_image_name)+'test'+fname+'.jpg')
            cv2.imwrite(corner_img_dir_path,img)
            corner_image_name = corner_image_name+1
            """
            传入所有图片各自角点的三维，二维坐标，相机标定
            每个图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组
            mtx,相机内参；dist，畸变系数；revcs，旋转矩阵；tveces，平移矩阵
            """
    print(len(objpoints))
    print(len(imgpoints))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
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
    print(camera_+'标定完成！总用时'+str(times)+'秒')
    print("===========================================")
    print('ret:', single_calibration_fs_test.getNode('ret').real())  # 正确
    print(camera_+'相机内参:', single_calibration_fs_test.getNode('mtx').mat())  # 正确  相机内参
    print(camera_+'相机畸变:', single_calibration_fs_test.getNode('dist').mat())  # 正确  畸变系数
    print(camera_+'旋转矩阵1:', single_calibration_fs_test.getNode('rvecs_1').mat())
    print(camera_+'平移矩阵1:', single_calibration_fs_test.getNode('tvecs_1').mat())
    print('===========================================')
    print(camera_+'标定效果评估:', single_calibration_fs_test.getNode('tot_error').real())
    print('compute_distance_path:', single_calibration_fs_test.getNode('compute_distance_path').string())
    print('corner_img_dir:', single_calibration_fs_test.getNode('corner_img_dir').string())

    ##file_names = check_up_txt_file(source_dir_path);  ##后续函数
    print('===========================================')

    ##print(file_names)

    print('===========================================')

    ##for file_name in file_names:
        ##input_name = file_name.split('/')[8].split('.')[0]

        # final_object[input_name]  has  list  , float ,  numpy.ndarray and  str  type
        #  we  must  according  the type  use  different  ways  to save the  object
        # todo 最后的存储的东西有很多数据类型，我们需要根据数据类型存储
        ##save_object(final_object[input_name], file_name)  ##后续函数
    ##return 1

"""
camera inner parameters
rotate  matrix 
transfer vector 
stereo calibration processing 

"""


def stereo_Calibration(flag):
    starttime = time.time()
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    # Prepare object points
    # todo  修改点的数量
    objp = np.zeros((13 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:13, 0:9].T.reshape(-1, 2)

    # Arrays to  store object points and image points from all images
    objpoints = []
    imgpointsR = []
    imgpointsL = []

    # Start calibration from the camera
    left_camera_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/')
    leftCamera = left_camera_path+flag+"_leftCamera_img_dir/"
    #left_camera_path = "/home/monkiki/PycharmProjects/stereo_vision/stereo_vision/checkboard_img_dir/leftCamera_img_dir/"
    right_camera_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/')
    rightCamera = right_camera_path+flag+"_rightCamera_img_dir/"
    #right_camera_path = "/home/monkiki/PycharmProjects/stereo_vision/stereo_vision/checkboard_img_dir/rightCamera_img_dir/"
    for i in os.listdir(rightCamera):

        if i.endswith('.jpg') or i.endswith('.png'):
            img_path = os.path.join(rightCamera, i)
            print(img_path)
            ChessImaR = cv2.imread(img_path)
            ChessImaR = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)
            # todo  修改点的数量
            retR, cornersR = cv2.findChessboardCorners(ChessImaR, (13, 9), None)
            # objpoints.append(objp)
            cornersR = cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)

    for i in os.listdir(leftCamera):

        if i.endswith('.jpg') or i.endswith('.png'):
            img_path = os.path.join(leftCamera, i)
            print(img_path)
            ChessImaL = cv2.imread(img_path)
            ChessImaL = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
            retL, cornersL = cv2.findChessboardCorners(ChessImaL, (13, 9), None)
            objpoints.append(objp)
            cornersL = cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)

    # determine the new values for different parameters
    # Right Side

    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)

    hR, wR = ChessImaR.shape[:2]

    # todo  消除畸变后的新相机矩阵
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    # Letf Side

    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)

    hL, wL = ChessImaL.shape[:2]

    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    print('--------------开始进行双目标定--------------')

    """
    calibrate the Cameras for Stereo

    """
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    # TODO  M is  inner matrix   d is distorted

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                               imgpointsL,
                                                               imgpointsR,
                                                               OmtxL,
                                                               distL,
                                                               OmtxR,
                                                               distR,
                                                               ChessImaR.shape[::-1],
                                                               criteria_stereo,
                                                               flags)
    rectify_scale = 0
    """
    R is rotation matrix  P is projection matrix 
    """

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale,
                                                      (0, 0))
    """
    create the pixel projection matrix of the rectified picture 生成像素映射矩阵
    cv2.CV_16SC2 this format enables us the programme to work faster 
    """
    # TODO  需要存储此数据 save  this  map to  object file
    # Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS,dLS,RL,PL,ChessImaR.shape[::-1],cv2.CV_16SC2)

    # Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)

    Left_Stereo_Map_0, Left_Stereo_Map_1 = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1],
                                                                       cv2.CV_16SC2)

    Right_Stereo_Map_0, Right_Stereo_Map_1 = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1],
                                                                         cv2.CV_16SC2)

    calibration_path = os.path.dirname(os.path.realpath(__file__)).replace('cameraCalibration','static/checkboard_img_dir/' +flag+'_'+'stereo_calibration.xml')
    #calibration_path = "/home/monkiki/PycharmProjects/try/stereo_calibration.xml"
    stereo_calibration_fs = cv2.FileStorage(calibration_path,cv2.FileStorage_WRITE)
    stereo_calibration_fs.write('mtx_stereo', MLS)
    stereo_calibration_fs.write('dts_stereo', dLS)
    stereo_calibration_fs.write('rective_stereo', Q)
    # stereo_calibration_fs.write('left_stereo_map0', Left_Stereo_Map[0])
    # stereo_calibration_fs.write('left_stereo_map1', Left_Stereo_Map[1])
    # stereo_calibration_fs.write('right_stereo_map0', Right_Stereo_Map[0])
    # stereo_calibration_fs.write('right_stereo_map1', Right_Stereo_Map[1])

    stereo_calibration_fs.write('left_stereo_map0', Left_Stereo_Map_0)  ##
    stereo_calibration_fs.write('left_stereo_map1', Left_Stereo_Map_1)  ##
    stereo_calibration_fs.write('right_stereo_map0', Right_Stereo_Map_0)  ##
    stereo_calibration_fs.write('right_stereo_map1',Right_Stereo_Map_1)  ##这里存了一堆图片值，是矫正后的x、y映射,需要注意的是，map0负责横向映射，map1负责纵向映射
    stereo_calibration_fs.write('R_matrix', R)
    stereo_calibration_fs.write('T_matrix', T)
    stereo_calibration_fs.write('fundamental_matrix', F)

    stereo_calibration_fs.release()
    endtime = time.time()
    times = endtime - starttime
    print("双目标定完成！用时" + str(times) + "秒")


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





if __name__ =='__main__':

    #path = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/checkboard_img_dir/leftCamera_img_dir/'
    #test_json = sig_calibration(path)
    #print(test_json)
    #find_camera_calibration_file(path)
    #check_up_txt_file(path)

    left_name  =  '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/5.21/2019-05-21-17-27-23/left.jpg'

    right_name = '/home/cx/PycharmProjects/stereo_vision/stereo_vision/static/zj_pictures/5.21/2019-05-21-17-27-23/right.jpg'

    #stereo_Calibration()
    stereo_Calibration_for_use(left_name, right_name)



























