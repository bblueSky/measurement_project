import sys
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import random
import time

def robotComunicate() :
    logger = modbus_tk.utils.create_logger(name = "console", record_format = "%(message)s")
    try:
        #Create the server
        server = modbus_tcp.TcpServer(address="0.0.0.0", port=8080)
        #server.set_timeout(5.0)
        logger.info("running...")
        logger.info("enter 'quit' for closing the server")
        server.start()
        slave_1 = server.add_slave(1)
        slave_1.add_block('A', cst.HOLDING_REGISTERS, 0, 10)
        slave_1.set_values('A', 0, 10*[123])
        numsOfcom = 0
        while True:
            slave_1.set_values('A', 0, 10*[random.randint(0,200)])  # 改变在地址0处的寄存器的值
            print("==========发送"+str(numsOfcom)+"次数据==========")
            numsOfcom+=1
            time.sleep(1)
    finally:
        server.stop()