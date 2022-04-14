import numpy as np
import cv2
from scipy import ndimage as ndi
from djitellopy import Tello
import time

from matplotlib import pyplot as plt

tello = Tello()
connected = False

#img = cv2.imread('test (7).jpg')

try:
    #tello.connect()
    raise ValueError('Not trying to connect to drone - uncomment line above')
    print("Connected to tello")
    connected = True
    # connect opencv to live video
except:
    print("Failed to connect to tello")


def find_lines (img):
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #smooth = ndi.filters.median_filter(gray, size=2)
    #cv2.imshow("SMOOTHED", smooth)
    #cv2.waitKey(0)
    #edges = smooth > 180 #180
    mask = cv2.inRange(img, lower_white, upper_white)
    img = cv2.bitwise_and(img, img, mask=mask)
    edges = cv2.Canny(img, 100, 200)
    """
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    """
    # draw_lines(edges,(255,255,0))
    # edges,1,np.pi/180, 200
    lines = cv2.HoughLines(edges.astype(np.uint8), .2, np.pi / 180, 40)  # edges.astype(np.uint8), .4, np.pi / 180, 120
    # lines = cv2.HoughLines(edges.astype(np.uint8), .1, np.pi / 180, 120)
    # formats lines as an array of rho theta pairs
    tmp = np.empty(shape=(1, 2))
    try:
        for line in range(len(lines)):
            tmp = np.vstack((tmp, lines[line][0]))
        tmp = np.delete(tmp, (0), axis=0)
    except:
        print("no lines detected")
    return tmp

def corner_dist():
    global connected
    # improve angle with accelerometer reading
    camera_angle = 0.349066
    if connected:
        elevation = tello.get_height()
    else:
        elevation = 100
    dist = elevation / np.tan(camera_angle)
    #print("Distance from intersection: " + str(dist))
    return dist


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


def group_lines(lines, itterations):
    for i in range(itterations):
        lines = group_similar(lines, 1 * (i % 2 == 0))
    return lines


def get_closest_line(lines, img):
    #print(img.shape)
    y0, x0, c = img.shape
    #print(x0,y0)
    draw_point([x0/2, y0/2], (255,0,255), img)
    distances = []
    # open
    for line in lines:
        x = np.cos(line[1]) * line[0]
        y = np.sin(line[1]) * line[0]
        dist = abs(np.cos(line[1]+np.pi/2)*(y - y0/2) - np.sin(line[1]+np.pi/2)*(x - x0/2))
        distances.append(dist)
        #print(dist)
    index_min = min(range(len(distances)), key=distances.__getitem__)
    closest_line = lines[index_min]
    return closest_line


def check_intersection(line1, line2):
    # Intersection min angle tolerance
    min_angle = 5*(np.pi/180)
    # Line format is [rho, theta]
    rho1 = line1[0]
    rho2 = line2[0]
    the1 = line1[1]
    the2 = line2[1]

    # A = [cos θ1  sin θ1]   b = |r1|   X = |x|
    #     [cos θ2  sin θ2]       |r2|       |y|
    a1 = np.array([[ np.cos(the1), np.sin(the1) ],
                   [ np.cos(the2), np.sin(the2) ]])
    b1 = np.array([[rho1], [rho2]])

    try:
        sol = np.linalg.solve(a1, b1).T[0]
        #sol_angle = np.arctan2(sol[1], sol[0])
        #angle = (np.pi/2 - (the1 - sol_angle)) + (np.pi/2 - (the2 - sol_angle))
        angle = np.pi - (max(the1, the2) - min(the1,the2))
        angle = np.abs(min(angle, np.pi - angle))
        print("Angle: " + str(angle*(180/np.pi)))

        if angle >= min_angle:
            return sol
        else:
            return False
    except np.linalg.LinAlgError:
        return False


def point_within_frame(point, img_in):
    imsize = img_in.shape
    xpos = point[0]
    ypos = point[1]
    if 0 <= xpos <= imsize[0]:
        if 0 <= ypos <= imsize[1]:
            return True
        else:
            return False
    else:
        return False


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
    cv2.circle(img, (pt_x, pt_y), radius=2, color=(color), thickness=2)


def draw_points(points, color, img):
    for point in points:
        draw_point(point, color, img)


def draw_line(line, color, img):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(color),2)


def draw_lines (lines, color, img):
    for line in lines:
        draw_line(line, color, img)

def get_line_angle(line):

    return angle

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

def fly_drone(lines, img):
    global connected
    # update elevation to keep it constant
    # for now to this:
    y_vel = 0

    vel_y = 10
    Px = 1
    Ptheta = 1
    x, y, c = img.shape
    x = x/2
    y = y/2

    des_theta = np.pi / 2
    des_x = x / 2
    line = get_closest_line(lines, img)
    pos_theta = line[1] - np.pi/2
    vel_theta = Ptheta + (des_theta - pos_theta)
    pos_x = get_line_x(line, [x, y])
    vel_x = Px * (des_x - pos_x)
    if connected:
        tello.send_rc_control(vel_x, vel_y, y_vel, vel_theta)
    else:
        #print("Vel x: " + str(vel_x))
        #print("Vel theta: " + str(vel_theta))
        #print("Line x: " + str(get_line_x(line, [x, y])))
        pass

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    lines = find_lines(frame)
    lines = group_lines(lines, np.clip((len(lines) * 10),0,1000))
    close_line = get_closest_line(lines, frame)

    intersections = find_intersections(lines, frame)
    #print(intersections)

    draw_lines(lines, (0, 0, 255), frame)
    draw_line(close_line, (0, 255, 0), frame)
    draw_points(intersections, (255, 0, 0), frame)

    fly_drone(lines, frame)
    # Show the result

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
