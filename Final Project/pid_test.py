from simple_pid import PID
from djitellopy import Tello

tello = Tello()

pid_z = PID(1, 0.1, 0.05, setpoint=100)
pid_z.output_limits = (-10, 10)
pid_z.sample_time = 0.01  # Update every 0.01 seconds

try:
    tello.connect()
    print("Connected to tello")
    connected = True
except:
    print("Failed to connect to tello")
    connected = False

while connected:
    z = tello.get_height()
    #z = tello.get_distance_tof()
    #print(z)
    #print(tello.get_battery())
    vel_z = pid_z(z)
    tello.send_rc_control(0, 0, vel_z, 0)