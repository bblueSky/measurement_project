import sys
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import random
import time
from stereo_vision.robotMachine import buildServer as bs

def robotComunicate() :
    bs.buildSlave(1)
    bs.slaveAdd_block(1,'A', cst.HOLDING_REGISTERS, 0, 1)
    bs.slaveSet_values(1,'A',0,0)
    numsOfcom = 0
    while numsOfcom<3:
        bs.slaveSet_values(1,'A',0,1)  # 改变在地址0处的寄存器的值
        print("==========发送"+str(numsOfcom+1)+"次数据==========")
        numsOfcom+=1
        time.sleep(1)
    bs.slaveSet_values(1,'A',0,2)