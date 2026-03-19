from geometry import detect_geometry
from blade_present import wait_for_blade
from common import log

from time import sleep
import serial
from enum import Enum
import numpy as np
from datetime import datetime

class State(Enum):
    CONNECT      = 0
    WAIT_START   = 1
    DETECT_BLADE = 2
    SEND_BLADE_DETECTED = 3
    WAIT_VISION  = 4
    VISION = 5

START = 'START'
START_VISION = 'START_VISION'
RECONNECT = "RECONNECT"
KNIFE_DETECTED = "KNIFE_DETECTED"
FAIL = "FAIL"

def open_port(debug=False):
    while True:
        try:
            ser = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
            log("Connected", debug=debug)
            return ser
        except serial.SerialException:
            log("Waiting for device...", debug=debug)
            sleep(1)

def send_array(ser):
    data = np.load("/home/chuddycholo/XCalibur-Vision-IK/knife_data.npz")

    tip_q1 = data['tip_q1']
    tip_q2 = data['tip_q2']
    yaw_indices = data['yaw_indices']
    ratios1 = data['ratios1']
    ratios2 = data['ratios2']

    N = np.int16(len(yaw_indices))

    log(f"N {N}")
    log(f"tip_q1 {tip_q1}")
    log(f"tip_q2 {tip_q2}")
    log(f"yaw_indices {yaw_indices}")
    log(f"ratios1 {ratios1}")

    block_1 = bytes()
    block_1 += (tip_q1.astype(np.float64).tobytes() + tip_q2.astype(np.float64).tobytes()[:3])

    block_2 = bytes()
    block_2 += (tip_q2.astype(np.float64).tobytes()[3:]+ N.astype(np.int16).tobytes())

    CHUNK_SIZE = 64

    ser.write(block_1)
    sleep(0.02)
    log("Sent block 1")

    ser.write(block_2)
    sleep(0.02)
    log("Sent block 2")

    sleep(1)

    for i in range(0, N, 8):
        block = bytes()
        block += (yaw_indices[i:i+8]).astype(np.float64).tobytes()
        ser.write(block)
        sleep(0.02)
        log(str(yaw_indices[i:i+8]))

    log("sent yaw")
    sleep(1)

    for i in range(0, N+1, 2):
        block = bytes()
        block += (ratios1[i:i+2]).astype(np.float64).tobytes()
        ser.write(block)
        sleep(0.02)

    log("sent ratios1")

    for i in range(0, N+1, 2):
        block = bytes()
        block += (ratios2[i:i+2]).astype(np.float64).tobytes()
        ser.write(block)
        sleep(0.02)

    log("sent ratios2")
    log("Done sending")

    return True

def main(debug = False):
    state = State.CONNECT
    ser = None

    with open("log.txt", "a") as f:
        f.write(f"\n--- Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    while True:
        if state == State.CONNECT:
            # ser = open_port(debug=debug)
            try:
                # ser.write(RECONNECT.encode('utf-8'))
                state = State.WAIT_START
            except serial.SerialException:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT

        elif state == State.WAIT_START:
            try:
                # line = ser.readline().decode().strip()
                line = START
                if START in line:
                    log("Got START", debug=debug)
                    state = State.DETECT_BLADE
            except serial.SerialException:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT

        elif state == State.DETECT_BLADE:
            result = wait_for_blade(debug=debug)
            if not result:
                try:
                    # ser.write(FAIL.encode('utf-8'))
                    log("Knife Detection Failed!", debug=debug)
                    state = State.WAIT_START
                except serial.SerialException:
                    log("Lost connection", debug=debug)
                    # ser.close()
                    state = State.CONNECT
                continue
            try:
                # ser.write(KNIFE_DETECTED.encode('utf-8'))
                log("Wrote knife detected!", debug=debug)
                state = State.WAIT_VISION
            except serial.SerialException:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT

        elif state == State.WAIT_VISION:
            try:
                # line = ser.readline().decode().strip()
                line = START_VISION
                if START_VISION in line:
                    log("Got START_VISION", debug=debug)
                    state = State.VISION
            except serial.SerialException:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT

        elif state == State.VISION:
            result = detect_geometry(debug=debug)
            # if we alread failed, don't proceed
            if not result:
                try:
                    # ser.write(FAIL.encode('utf-8'))
                    log("Profile Detection Failed!", debug=debug)
                    state = State.WAIT_START
                except serial.SerialException:
                    log("Lost connection", debug=debug)
                    # ser.close()
                    state = State.CONNECT
                continue

            # otherwise try to send data
            tip_q1 = result[0]
            tip_q2 = result[1]
            yaw_indices = result[2]
            ratios1 = result[3]
            ratios2 = result[4]
        
            N = np.int16(len(yaw_indices))
        
            # log(f"N {N}", debug=debug)
            # log(f"tip_q1 {tip_q1}", debug=debug)
            # log(f"tip_q2 {tip_q2}", debug=debug)
            # log(f"yaw_indices {yaw_indices}", debug=debug)
            # log(f"ratios1 {ratios1}", debug=debug)
        
            block_1 = bytes()
            block_1 += (tip_q1.astype(np.float64).tobytes() + tip_q2.astype(np.float64).tobytes()[:3])
        
            block_2 = bytes()
            block_2 += (tip_q2.astype(np.float64).tobytes()[3:]+ N.astype(np.int16).tobytes())

            try:
                # ser.write(block_1)
                sleep(0.02)
                log("Sent block 1", debug=debug)
            except serial.SerialException:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT
                continue
        
            try:
                # ser.write(block_2)
                log("Sent block 2", debug=debug)
                sleep(1)
            except serial.SerialException:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT
                continue

            try:
                for i in range(0, N, 8):
                    block = bytes()
                    block += (yaw_indices[i:i+8]).astype(np.float64).tobytes()
                    # ser.write(block)
                    sleep(0.02)
                log("sent yaw", debug=debug)
                sleep(1)
            except:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT
                continue
            
            try:
                for i in range(0, N-1, 2):
                    block = bytes()
                    block += (ratios1[i:i+2]).astype(np.float64).tobytes()
                    # ser.write(block)
                    sleep(0.02)
                log("sent ratios1", debug=debug)
            except:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT
                continue
            
            try:
                for i in range(0, N-1, 2):
                    block = bytes()
                    block += (ratios2[i:i+2]).astype(np.float64).tobytes()
                    # ser.write(block)
                    sleep(0.02)
                log("sent ratios2", debug=debug)
            except:
                log("Lost connection", debug=debug)
                # ser.close()
                state = State.CONNECT
                continue

            log("Done sending", debug=debug)

            state = State.WAIT_START


if __name__ == "__main__":
    main(debug=True)