import serial
import time

ser = serial.Serial('COM4', 115200, timeout=10)
time.sleep(1)          # give Pi time to open port

ser.write(b'12345678 START_PRESSED!\n')
ser.flush()

time.sleep(1)          # ensure transmission finishes
ser.close()