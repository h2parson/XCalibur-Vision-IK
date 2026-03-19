from geometry import detect_geometry
from blade_present import wait_for_blade

from time import sleep

print("starting program")
wait_for_blade()
sleep(2)
print("beginning geometry detection")
blade_profile = detect_geometry()