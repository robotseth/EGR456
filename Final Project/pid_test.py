from simple_pid import PID
from djitellopy import Tello
import numpy as np
import time

tello = Tello()

pid_z = PID(3, 0.1, 0.1, setpoint=100)
pid_z.output_limits = (-20, 20)
pid_z.sample_time = 0.01  # Update every 0.01 seconds

try:
    tello.connect()
    print("Connected to tello")
    connected = True
    tello.takeoff()
except:
    print("Failed to connect to tello")
    connected = False

while connected:
    state = tello.get_current_state()
    print(state)
    if state is not None:
        connected = True
    else:
        connected = False
    #z = tello.get_height()
    z = state['tof']
    z = np.clip(z, 0, 200)

    print(z)
    time.sleep(.1)
    #print(tello.get_battery())
    vel_z = pid_z(z)
    tello.send_rc_control(0, 0, int(vel_z), 0)
print("connection lost")