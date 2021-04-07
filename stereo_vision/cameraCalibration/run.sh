#!/bin/bash
sudo  chmod -R 777 /dev/bus/usb
sudo  sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
