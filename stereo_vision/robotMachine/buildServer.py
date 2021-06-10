import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import random
import time


def __init__(ad,pt):
    """
    设置服务器地址、端口号
    启动服务器
    设置从机集合
    """
    global __logger__
    global __server__
    global __slaveList__
    __logger__ = modbus_tk.utils.create_logger(name = "console", record_format = "%(message)s")
    __server__ = modbus_tcp.TcpServer(address=ad, port=pt)
    __logger__.info("running...")
    __logger__.info("enter 'quit' for closing the server")
    __server__.start()
    __slaveList__ = list()


def buildSlave(index):
    """
    设置从机号 ##按照你们的习惯默认是从1开始，每多设置一个从机号就依次加一
    """
    __slaveList__.append(__server__.add_slave(index))

def slaveAdd_block(index,sname,cls,register,bname):
    """
    :param index:    从机号
    :param sname:    block名
    :param cls:      类型参数
    :param register: 寄存器地址
    :param bname:    block号

    """
    __slaveList__[index-1].add_block(sname,cls,register, bname)


def slaveSet_values(index,sname,register,value):
    """
    :param index:    从机号
    :param sname:    block名
    :param register: 寄存器地址
    :param value:    要设置的值
    """
    __slaveList__[index - 1].set_values(sname,register,value)



def stopServer():
    __server__.stop()