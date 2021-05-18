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










