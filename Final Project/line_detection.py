import numpy as np
import cv2
from scipy import ndimage as ndi
from djitellopy import Tello

tello = Tello()
connected = False

img = cv2.imread('test (6).jpg')

try:
    #tello.connect()
    raise ValueError('Not trying to connect to drone - uncomment line above')
    print("Connected to tello")
    connected = True
    # connect opencv to live video
except:
    print("Failed to connect to tello")


def find_lines ():
    global img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = ndi.filters.median_filter(gray, size=2)
    cv2.imshow("SMOOTHED", smooth)
    cv2.waitKey(0)
    edges = smooth > 200 #180
    #draw_lines(edges,(255,255,0))
    # edges,1,np.pi/180, 200
    lines = cv2.HoughLines(edges.astype(np.uint8), .3, np.pi / 180, 100) #edges.astype(np.uint8), .4, np.pi / 180, 120
    #lines = cv2.HoughLines(edges.astype(np.uint8), .1, np.pi / 180, 120)
    # formats lines as an array of rho theta pairs
    tmp = np.empty(shape=(1, 2))
    for line in range(len(lines)):
        tmp = np.vstack((tmp, lines[line][0]))
    tmp = np.delete(tmp, (0), axis=0)
    return tmp


def corner_dist ():
    global connected
    # improve angle with accelerometer reading
    camera_angle = 0.349066
    if connected:
        elevation = tello.get_height()
    else:
        elevation = 100
    dist = elevation / np.tan(camera_angle)
    print("Distance from intersection: " + str(dist))
    return dist


def group_similar (data, axis):
    rho_threshold = 10
    theta_threshold = .1
    data = data[data[:, axis].argsort()] # sorts array
    tmp = np.zeros(shape=(1,2))
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


def group_lines (lines, itterations):
    for i in range(itterations):
        lines = group_similar(lines, 1 * (i % 2 == 0))
    return lines


def get_closest_line (lines):
    #print(img.shape)
    x0, y0, c = img.shape
    distances = []
    # open
    for line in lines:
        x = np.cos(line[1]) * line[0]
        y = np.sin(line[1]) * line[0]
        dist = abs(np.sin(y - y0/2) - np.sin(x - x0/2))
        distances.append(dist)
        print(dist)
    index_min = min(range(len(distances)), key=distances.__getitem__)
    closest_line = lines[index_min]
    return closest_line


def check_intersection(line1, line2):
    # Line format is [rho, theta]
    rho1 = line1[0]
    rho2 = line2[0]
    the1 = line1[1]
    the2 = line2[1]

    # A = [cos θ1  sin θ1]   b = |r1|   X = |x|
    #     [cos θ2  sin θ2]       |r2|       |y|
    a1 = np.array([[ np.cos(the1), np.sin(the1) ],[ np.cos(the2), np.sin(the2) ]])
    b1 = np.array([[rho1],[rho2]])

    # TODO: Figure out why it finds intersections where there are no lines
    try:
        sol = np.linalg.solve(a1, b1).T[0]
        return sol
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


def find_intersections(lines):
    global img
    # Iterate through half of the lines
    found_pts = []
    # TODO: Iterate over the lines in a smarter way, this is stupid and only for testing
    for i in range(len(lines)-1):
        pt = check_intersection(lines[i], lines[i+1])
        #print(pt)
        if pt is not False:
            if point_within_frame(pt, img):
                found_pts.append(pt)
    return found_pts


def draw_point (point, color):
    pt_x = int(np.round(point[0]))
    pt_y = int(np.round(point[1]))
    cv2.circle(img, (pt_x, pt_y), radius=2, color=(color), thickness=2)


def draw_points (points, color):
    for point in points:
        draw_point(point, color)


def draw_line (line, color):
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


def draw_lines (lines, color):
    for line in lines:
        draw_line(line, color)

lines = find_lines()
lines = group_lines(lines, 400)
close_line = get_closest_line(lines)

intersections = find_intersections(lines)

draw_lines(lines, (0,0,255))
draw_line(close_line, (0,255,0))
draw_points(intersections, (255,0,0))

# Show the result
cv2.imshow("Line Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()