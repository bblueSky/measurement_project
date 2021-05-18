#encoding:utf-8

from flask  import render_template, request
from stereo_vision.robotMachine import robotMachine
import time
import json
import datetime
import os




@robotMachine.route('/')
def index():

    return render_template('robot_machine/robot_machine.html')










