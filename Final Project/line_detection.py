import numpy as np
import cv2
from scipy import ndimage as ndi
from djitellopy import Tello
from simple_pid import PID
import time

from matplotlib import pyplot as plt

def intializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 10
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

img_static = cv2.imread('test (7).jpg')
""" initializes PID objects for the fly_drone() function to use """
# pid_x.sample_time = 0.01  # Update every 0.01 seconds
# the line above can be used to set the sample time, but it is assumed that the frame time will be consistent
pid_x = PID(0.05, 0.01, 0, setpoint=0)
pid_x.output_limits = (-15, 15)
#pid_theta = PID(4, 0.1, 0.1, setpoint=int(np.pi / 2))
pid_theta = PID(30, 0.5, 0, setpoint=0)
pid_theta.output_limits = (-40, 40)
pid_z = PID(1, 0.1, 0.1, setpoint=80)
pid_z.output_limits = (-20, 20)
pid_running = False
advance_on_line = False
pid_x_running = True
pid_theta_running = True


try:
    tello = intializeTello()
    #raise ValueError('Not trying to connect to drone - uncomment line above')
    print("Connected to tello")
    connected = True
    # connect opencv to live video
except:
    print("Failed to connect to tello")
    connected = False


def find_lines (img):
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    lower_blue = np.array([80, 0, 0])
    upper_blue = np.array([255, 150, 80])
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #smooth = ndi.filters.median_filter(gray, size=2)
    #cv2.imshow("SMOOTHED", smooth)
    #cv2.waitKey(0)
    #edges = smooth > 180 #180
    y, x, c = img.shape
    mask_0 = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask_0, (0, y), (x, 0), 255, -1)
    mask_1 = cv2.inRange(img, lower_white, upper_white)
    mask = mask_0 & mask_1
    cv2.imshow('mask 0', mask_0)
    cv2.imshow('mask 1', mask_1)
    img_0 = cv2.bitwise_and(img, img, mask=mask_0)
    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('img 0', img_0)
    cv2.imshow('img 1', img)

    edges = cv2.Canny(img, 100, 200)
    cv2.imshow('edges', edges)
    """
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    """
    # draw_lines(edges,(255,255,0))
    # edges,1,np.pi/180, 200
    lines = cv2.HoughLines(edges.astype(np.uint8), .2, np.pi / 180, 30)  # edges.astype(np.uint8), .4, np.pi / 180, 120
    #circles = cv2.HoughCircles(edges.astype(np.uint8), .2, np.pi / 180, 40)
    # lines = cv2.HoughLines(edges.astype(np.uint8), .1, np.pi / 180, 120)
    # formats lines as an array of rho theta pairs
    tmp = np.empty(shape=(1, 2))
    try:
        for line in range(len(lines)):
            tmp = np.vstack((tmp, lines[line][0]))
        tmp = np.delete(tmp, (0), axis=0)
    except:
        print("no lines detected")
        tello.for_back_velocity = 0
        tello.left_right_velocity = 0
        tello.up_down_velocity = 0
        tello.yaw_velocity = 0
    return tmp


# gets the distance of the drone to an intersection
# assumes the intersection is straight in front of the drone and it is centered in the camera frame
def corner_dist():
    global connected
    # improve angle with accelerometer reading
    camera_angle = 0.349066
    if connected:
        elevation = tello.get_height()
    else:
        elevation = 10
    dist = int(elevation / np.tan(camera_angle))
    #print("Distance from intersection: " + str(dist))
    return dist


# combines lines that are close to each other in a very inefficient way
def group_similar(data, axis):
    rho_threshold = 15
    theta_threshold = .1
    data = data[data[:, axis].argsort()]  # sorts array
    tmp = np.zeros(shape=(1, 2))
    rows, columns = data.shape
    for i in range(rows - 1):
        if abs(data[i][0] - data[i+1][0]) <= rho_threshold and abs(data[i][1] - data[i+1][1]) <= theta_threshold:
            # tmp.append([data[i], data[i+1]])
            # average values and append to array
            b = np.array([data[i][0], data[i][1], data[i+1][0], data[i+1][1]]).reshape(2,2)
            c = b.mean(axis=0)
            tmp = np.vstack((tmp,c))
        else:
            # append unaltered row
            c = np.array([data[i][0], data[i][1]])
            tmp = np.vstack((tmp, c))
    c = np.array([data[-1][0], data[-1][1]])
    tmp = np.vstack((tmp, c))
    tmp = np.delete(tmp, (0), axis=0)
    return tmp


# calls the group similar function a number of times to group all the lines
def group_lines(lines, iterations):
    for i in range(iterations):
        lines = group_similar(lines, 1 * (i % 2 == 0))
    return lines


# finds and returns the line closest to the center of the image frame
# length seems fine...maybe
# angle is off
def get_closest_line(lines, img):
    #print(img.shape)
    y0, x0, c = img.shape
    #print(x0,y0)
    draw_point([x0/2, y0/2], (255,0,255), img)
    distances = []
    # open
    for line in lines:
        angle_range = 45*(np.pi/180)
        if (0 < line[1] < (angle_range/2)) or ((np.pi - angle_range/2) < line[1] < np.pi):
            y = np.sin(line[1]) * line[0]
            x = np.cos(line[1]) * line[0]
            dist = abs(np.cos(line[1]+np.pi/2)*(y - y0/2) - np.sin(line[1]+np.pi/2)*(x - x0/2))
            distances.append(dist)
        #print(dist)
    try:
        index_min = min(range(len(distances)), key=distances.__getitem__)
        closest_line = lines[index_min]
        return closest_line
    except:
        print("no vertical lines found, returning centered vertical line")
        return [x0/2, 0]


# checks if the intersection is between two lines that are almost parallel
# if they are close to parallel, ignore the intersection
def check_intersection(line1, line2):
    # Intersection min angle tolerance
    min_angle = 5*(np.pi/180)
    # Line format is [rho, theta]
    rho1 = line1[0]
    rho2 = line2[0]
    the1 = line1[1]
    the2 = line2[1]

    # A = [cos ??1  sin ??1]   b = |r1|   X = |x|
    #     [cos ??2  sin ??2]       |r2|       |y|
    a1 = np.array([[ np.cos(the1), np.sin(the1) ],
                   [ np.cos(the2), np.sin(the2) ]])
    b1 = np.array([[rho1], [rho2]])

    try:
        sol = np.linalg.solve(a1, b1).T[0]
        #sol_angle = np.arctan2(sol[1], sol[0])
        #angle = (np.pi/2 - (the1 - sol_angle)) + (np.pi/2 - (the2 - sol_angle))
        angle = np.pi - (max(the1, the2) - min(the1,the2))
        angle = np.abs(min(angle, np.pi - angle))
        #print("Angle: " + str(angle*(180/np.pi)))

        if angle >= min_angle:
            return sol
        else:
            return False
    except np.linalg.LinAlgError:
        return False


# checks if the intersection is within the camera frame
# ignore intersections outside the frame
def point_within_frame(point, img_in):
    imsize = img_in.shape
    xpos = point[0]
    ypos = point[1]
    if 0 <= xpos <= imsize[1]:
        if 0 <= ypos <= imsize[0]:
            return True
        else:
            return False
    else:
        return False


# finds the intersections between the lines found in the image and returns them
def find_intersections(lines, img):
    # Iterate through half of the lines
    found_pts = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = check_intersection(lines[i], lines[j])
            #print(pt)
            if pt is not False:
                if point_within_frame(pt, img):
                    if not found_pts:
                        found_pts = [pt]
                    else:
                        dist = []
                        for point in found_pts:
                            dist = np.sum(np.square(pt - point))
                        if min([dist, 10000]) > 10:
                            found_pts.append(pt)
    return found_pts


def draw_point(point, color, img):
    pt_x = int(np.round(point[0]))
    pt_y = int(np.round(point[1]))
    cv2.circle(img, [pt_x, pt_y], radius=2, color=(color), thickness=2)


def draw_points(points, color, img):
    for point in points:
        draw_point(point, color, img)


def draw_line(line, color, img):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (color), 2)


def draw_lines(lines, color, img):
    #lines = np.clip(lines, -2147483648, 2147483648)
    try:
        for line in lines:
            draw_line(line, color, img)
    except:
        print("error drawing lines")
        print("Type of array " + str(type(lines[0])))
        print("Array " + str(lines[0]))
        print("Type of first element " + str(type(lines[0])))


# similar to the draw line function except it the lines are not a set long length
def draw_line_segment(point1,point2,color,img):
    x1, y1 = point1
    x2, y2 = point2
    line_thickness = 2
    cv2.line(img, (x1, y1), (x2, y2), color, thickness=line_thickness)


# converts polar to cartesian coordinates
def pol2cart(rho, phi):
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return (x, y)


# finds the x_displacement of the line relative to the center of the frame
def get_line_x(line, img):
    y0, x0, c = img.shape
    x0 = int(x0/2)
    y0 = int(y0/2)
    rho_0 = np.sqrt(x0 ** 2 + y0 ** 2)
    phi_0 = np.arctan2(y0, x0)
    # center of the image = rho_0, phi_0
    rho_1 = line[0]
    phi_1 = line[1]
    rho_p = rho_1 - rho_0 * np.cos(abs(phi_0-phi_1)) #np.cos(abs(phi_0-phi_1))
    line_x, line_y = pol2cart(rho_p, phi_1) # pol2cart(rho_p, np.pi/2 - phi_1)
    line_x = int(line_x + x0)
    line_y = int(line_y + y0)
    #print(line_x)

    x_disp = int(rho_p * np.cos(phi_1))

    try:
        draw_line_segment([x0, y0], [line_x, line_y], (255,255,0), img)
        draw_line_segment([x0, y0], [x_disp + x0, y0], (200, 180, 80), img)
    except:
        print("Error drawing line segment")
    #theta = line[1] - (np.pi + line[1]) * (line[0] < 0)
    #print(theta)
    return x_disp


def detect_center_intersection(intersections, center_size, img):
    centered = False
    y0, x0, c = img.shape
    x0 = int(x0/2)
    y0 = int(y0/2)
    for intersection in intersections:
        if np.sqrt((intersection[0] - x0) ** 2 + (intersection[1] - y0) ** 2) < center_size:
            centered = True
            print("Centered intersection found!!!")
            break
    return centered


def get_angle(line_1, line_2):
    angle = line_1[1] - line_2[1]
    return angle
"""
def get_line_x(line, center):
    x0, y0 = center
    x = np.cos(line[1]) * line[0]
    y = np.sin(line[1]) * line[0]
    #rho = line[0] * (-1 * (line[0] < 0))
    dist = abs(np.cos(line[1] + np.pi / 2) * (y - y0 / 2) - np.sin(line[1] + np.pi / 2) * (x - x0 / 2))

    theta = line[1] - (np.pi + line[1]) * (line[0] < 0)
    #print(theta)
    print(line[0])
    dist_x = np.cos(theta) * dist
    return dist_x
"""

# main drone control function that is called regularly
def fly_drone(lines, intersections, img):
    global connected
    # update elevation to keep it constant
    # for now to this:
    state = tello.get_current_state()

    if connected:
        z = state['tof']
    else:
        z = 80

    deg = 90 # amount to rotate after seeing a corner
    # this would ideally change depending on the angle between the lines at the intersection
    # this angle in not easy to get without making many changes, however
    # for now, just leave it as a small value that we may need to change manually for each shape

    vel_x = 0
    vel_y = 10
    vel_z = 0
    vel_theta = 0
    y, x, c = img.shape
    x0 = x/2

    # if an intersection is detected at the center of the frame, move above it
    if detect_center_intersection(intersections, 20, frame) and connected:
        #print("moving forward towards intersection")
        #dist = corner_dist()
        #tello.move_forward(int(dist))
        print("rotating at intersection")
        tello.send_rc_control(0, 0, 0, 0)
        tello.rotate_counter_clockwise(deg)
        tello.send_rc_control(0, 10, 0, 0)
    else: # if an intersection is not centered on the frame, use line-based P control
        line = get_closest_line(lines, img)
        pos_theta = line[1]
        if pos_theta >= int(np.pi/2):
            pos_theta = pos_theta - int(np.pi)

        vel_theta = int(pid_theta(pos_theta))
        pos_x = get_line_x(line, frame)
        vel_x = int(pid_x(-pos_x))
        #vel_z = int(pid_z(z))
        if connected:
            if not advance_on_line:
                vel_y = 0

            if not pid_x_running:
                vel_x = 0

            if not pid_theta_running:
                vel_theta = 0

            if pid_running:
                tello.send_rc_control(vel_x, vel_y, vel_z, -vel_theta)

            #msg = f'Pos X is {pos_x} and pos theta is {pos_theta}.'
            #print(msg)
            msg = f'x = {pos_x} | theta = {round(pos_theta, 3)}'
            cv2.putText(img=img, text=msg, org=(10, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 0),thickness=1)

            #msg = f'PID X is {vel_x} {"right" if vel_x >= 0 else "left"} and PID theta is {vel_theta} {"cw" if vel_theta >= 0 else "ccw"}.'
            msg = f'V_x = {vel_x} | V_theta = {vel_theta}'
            cv2.putText(img=img, text=msg, org=(10, 62), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 0),thickness=1)
            #print(msg)

        else:
            #print("Vel x: " + str(vel_x))
            #print("Vel theta: " + str(vel_theta))
            #print("Line x: " + str(get_line_x(line, [x, y])))
            pass



#cap = cv2.VideoCapture(0)

def telloGetFrame(tello, w, h):
    frame = tello.get_frame_read()
    frame = frame.frame
    img = cv2.resize(frame, (w, h))
    return img
"""
if not cap.isOpened():
    print("Cannot open camera")
    exit()
"""

tello.takeoff()
tello.move_down(20)
#tello.move_down(93)

pid_x.reset()
pid_theta.reset()
pid_z.reset()

global msg
msg = " "

while True:
    # Capture frame-by-frame
    frame = telloGetFrame(tello, 640, 480)
    #ret, frame = cap.read()
    # Mirror frame on x-axis to account for drone-mounted mirror. Comment out for normal use
    frame = cv2.flip(frame, 0)
    # if frame is read correctly ret is True
    """
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    """
    # finds lines
    lines = find_lines(frame)
    lines = group_lines(lines, np.clip((len(lines) * 10), 0, 1000))
    close_line = get_closest_line(lines, frame)

    # visualize lines and intersections
    intersections = find_intersections(lines, frame)
    draw_lines(lines, (0, 0, 255), frame)
    try:
        draw_line(close_line, (0, 255, 0), frame)
    except:
        print("error drawing close line")
    draw_points(intersections, (255, 0, 0), frame)

    # control the drone
    fly_drone(lines, intersections, frame)

    # display lines
    x, y, c = frame.shape
    #draw_lines(np.array([[10,0],[10,np.pi/2],[x-10,np.pi/2],[y-10,0]]), (255,255,0), frame)
    #draw_point([0,0], [255,255,0], frame)
    #draw_point([0, 10], [255, 255, 0], frame)

    keyPressed = cv2.waitKey(1)

    # Keyboard commands
    # p = quit program
    # w, a, s, d, q, e = Standard controls
    # spacebar = stop movement in rc mode
    # t = Increase x p gain
    # r = Decrease x P gain
    # v = Increase theta P gain
    # c = Decrease theta P gain
    # f = Reset PID values
    # g = Toggle PID running
    # 2 = Toggle line advancing
    # 3 = Toggle x pid
    # 4 = Toggle theta pid

    if keyPressed == ord('p'):
        # Quit the program
        break
    elif keyPressed == ord('t'):
        # Increase x P gain
        pid_x.Kp += 0.05
        msg = f"x P gain up to {pid_x.Kp}"
        print(msg)
    elif keyPressed == ord('r'):
        # Decrease x P gain
        pid_x.Kp -= 0.05
        msg = f"x P gain down to {pid_x.Kp}"
        print(msg)
    elif keyPressed == ord('v'):
        # Increase theta P gain
        pid_theta.Kp += 1
        msg = f"Theta P gain up to {pid_theta.Kp}"
        print(msg)
    elif keyPressed == ord('c'):
        # Decrease theta P gain
        pid_theta.Kp -= 1
        msg = f"Theta P gain down to {pid_theta.Kp}"
        print(msg)
    elif keyPressed == ord('f'):
        # Reset PID values
        pid_x.reset()
        pid_theta.reset()
        tello.send_rc_control(0, 0, 0, 0)
        msg = "Reset PID"
        print(msg)
    elif keyPressed == ord('g'):
        # Toggle PID running
        pid_running = not pid_running
        msg = f"Turned PID {'on' if pid_running else 'off'}"
        print(msg)
    elif keyPressed == ord('2'):
        # Toggle line advancing
        advance_on_line = not advance_on_line
        msg = f"Turned line adv. {'on' if advance_on_line else 'off'}"
        print(msg)
    elif keyPressed == ord('3'):
        # Toggle x pid
        pid_x_running = not pid_x_running
        msg = f"Turned pid x {'on' if pid_x_running else 'off'}"
        print(msg)
    elif keyPressed == ord('4'):
        # Toggle theta pid
        pid_theta_running = not pid_theta_running
        msg = f"Turned pid theta {'on' if pid_theta_running else 'off'}"
        print(msg)


    if not pid_running:
        if keyPressed == ord('w'):
            msg = "Moving forward"
            print(msg)
            tello.send_rc_control(0, 10, 0, 0)
        elif keyPressed == ord('a'):
            msg = "Moving left"
            print(msg)
            tello.send_rc_control(-10, 0, 0, 0)
        elif keyPressed == ord('s'):
            msg = "Moving back"
            print(msg)
            tello.send_rc_control(0, -10, 0, 0)
        elif keyPressed == ord('d'):
            msg = "Moving right"
            print(msg)
            tello.send_rc_control(10, 0, 0, 0)
        elif keyPressed == ord('q'):
            msg = "Rotating ccw"
            print(msg)
            tello.send_rc_control(0, 0, 0, -10)
        elif keyPressed == ord('e'):
            msg = "Rotating cw"
            print(msg)
            tello.send_rc_control(0, 0, 0, 10)
        elif keyPressed == ord(' '):
            msg = "Holding still"
            #print(msg)
            tello.send_rc_control(0, 0, 0, 0)

    cv2.putText(img=frame, text=msg, org=(10, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 0),thickness=1)

    # Display the resulting frame
    cv2.imshow('frame', frame)


# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()
