import numpy as np
import cv2
from djitellopy import Tello

lower_blue = np.array([40, 0, 0])
upper_blue = np.array([255, 90, 40])
cap = cv2.VideoCapture(1)

def intializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

def telloGetFrame(tello, w, h):
    frame = tello.get_frame_read()
    frame = frame.frame
    img = cv2.resize(frame, (w, h))
    return img

tello = intializeTello()

while True:
    #ret, img = cap.read()
    img = telloGetFrame(tello, 640, 480)
    mask = cv2.inRange(img, lower_blue, upper_blue)
    masked = cv2.bitwise_and(img, img, mask=mask)
    not_masked = cv2.bitwise_not(img, img, mask=mask)
    cv2.imshow('masked', masked)
    #cv2.imshow('not_masked', not_masked)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()