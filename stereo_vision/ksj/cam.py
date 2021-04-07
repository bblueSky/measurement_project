# coding:utf-8
import datetime
import time
import numpy as np
import cv2
from ctypes import *
import pdb;


class Ksjcam:

    def __init__(self):
        print("init")
        self.initRes = self.KsjInit()
        self.libKsj = self.initRes[0]
        self.camNub = self.initRes[1]
        self.bufList = self.CreateBuf(self.libKsj, self.camNub)

    def read(self, index):
        nWidth = c_int()
        nHeight = c_int()
        nBitCount = c_int()
        self.libKsj.KSJ_CaptureGetSizeEx(index, byref(nWidth), byref(nHeight), byref(nBitCount))

        return self.CapturData(index, self.bufList[index], nWidth.value, nHeight.value, nBitCount.value >> 3)

    def CapturData(self, nIndex, cBuf, nHeight, nWidth, nChannelNum):

        if nChannelNum == 1:
            pdb.set_trace()
            retValue = self.libKsj.KSJ_CaptureRawData(nIndex, cBuf)

        if nChannelNum > 1:
            retValue = self.libKsj.KSJ_CaptureRgbData(nIndex, cBuf)

        if retValue != 0:
            print("capture error code %d" % (retValue))
        #    check_buf_data(cBuf)

        #    exptime = c_float()
        #    libKsj.KSJ_ExposureTimeGet(nIndex,byref(exptime))

        #    print("exptime = %f"%(exptime.value))

        nparr = np.fromstring(cBuf, np.uint8).reshape(nWidth, nHeight, nChannelNum);
        #    nparr = np.fromstring(cBuf,np.uint8).reshape(nWidth,nHeight,nChannelNum );

        return nparr

    def CapturDataLoop(self, nIndex, pDataBuf, nWidth, nHeight):
        nThreadFlag = 1;
        channelnum = 3
        nFrameCount = 0

        cv2.namedWindow("test" + str(0), cv2.WINDOW_AUTOSIZE)

        while nThreadFlag > 0:

            image = self.CapturData(nIndex, pDataBuf, nHeight, nWidth, int(channelnum))
            cv2.imshow("test" + str(nIndex), image)
            cv2.waitKey(1)

            if nFrameCount == 0:
                nTimeStart = datetime.datetime.now()

            if nFrameCount == 99:
                nTimeStop = datetime.datetime.now()
                deltt = (nTimeStop - nTimeStart)
                print("cam   %d  type %d  nSerials %d  fps  %f" % (
                    nIndex, usDeviceType.value, nSerials.value, 100 / (deltt.total_seconds())))
                global g_recordflag
                g_recordflag = 0
                vwriter1.release()

                nFrameCount = -1

            nFrameCount = nFrameCount + 1
            time.sleep(0.001)
        print("thread quit")

    def KsjInit(self):
        libKsj = cdll.LoadLibrary('libksjapi.so')
        libKsj.KSJ_Init()
        camCount = libKsj.KSJ_DeviceGetCount()
        print(camCount)
        if camCount < 1:
            print("no cam fount")
        else:
            print("cam found")
        return libKsj, camCount

    def CreateBuf(self, libKsj, num):
        buflist = []
        nWidth = c_int()
        nHeight = c_int()
        nBitCount = c_int()
        for i in range(0, num):
            libKsj.KSJ_CaptureGetSizeEx(i, byref(nWidth), byref(nHeight), byref(nBitCount))
            print(nWidth)
            print(nHeight)
            print(nBitCount)
            nbufferSize = nWidth.value * nHeight.value * nBitCount.value / 8
            pRawData = create_string_buffer(int(nbufferSize))
            buflist.append(pRawData)

        return buflist

    def SetExptime(self, index, exp_ms):
        self.libKsj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
        self.libKsj.KSJ_ExposureTimeSet(index, exp_ms);

    def SetTriggerMode(self, index, mode):
        self.libKsj.KSJ_TriggerModeSet.argtypes = (c_int, c_int)
        self.libKsj.KSJ_TriggerModeSet(index, mode);

    def SetFixedFrameRateEx(self, index, rate):
        self.libKsj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
        self.libKsj.KSJ_SetFixedFrameRateEx(index, rate);
