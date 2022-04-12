import numpy as np
import cv2
from scipy import ndimage as ndi
from collections import defaultdict

img = cv2.imread('test1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = ndi.filters.median_filter(gray, size=2)

cv2.imshow("SMOOTHED", smooth)
cv2.waitKey(0)

edges = smooth > 180

lines = cv2.HoughLines(edges.astype(np.uint8), .4, np.pi/180, 120)
print(lines)

# formats lines as an array of rho theta pairs
tmp = np.empty(shape=(1,2))
for line in range(len(lines)):
    tmp = np.vstack((tmp,lines[line][0]))
tmp = np.delete(tmp, (0), axis=0)
lines = tmp

rho_threshold = 5
theta_threshold = .1

# def corner_dist ():
#     camera_angle = 20
#     elevation = 1 # read sensor for real elevation
    

def group_similar (data, axis):
    data = data[data[:, axis].argsort()]# sorts array by theta
    tmp = np.zeros(shape=(1,2))
    #print(tmp)
    rows, columns = data.shape
    for i in range(rows - 1):
        if abs(data[i][0] - data[i+1][0]) <= rho_threshold and abs(data[i][1] - data[i+1][1]) <= theta_threshold:
            # tmp.append([data[i], data[i+1]])
            # average values and append to array
            b = np.array([data[i][0], data[i][1], data[i+1][0], data[i+1][1]]).reshape(2,2)
            #print(b)
            c = b.mean(axis=0)
            #print(c)
            tmp = np.vstack((tmp,c))
            #print("if triggered")
        else:
            # append unaltered row
            c = np.array([data[i][0], data[i][1]])
            tmp = np.vstack((tmp, c))
    c = np.array([data[-1][0], data[-1][1]])
    tmp = np.vstack((tmp, c))
    tmp = np.delete(tmp, (0), axis=0)
    #print(tmp)
    return tmp

def group_lines (lines, itterations):
    for i in range(itterations):
        lines = group_similar(lines, 1 * (i % 2 == 0))
    return lines


lines = group_lines(lines, 40)


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


intersections = find_intersections(lines)

for i in range(len(lines)):
    rho,theta = lines[i]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

for pt in intersections:
    pt_x = int(np.round(pt[0]))
    pt_y = int(np.round(pt[1]))
    print(pt_x, ",", pt_y)
    cv2.circle(img, (pt_x, pt_y), radius=2, color=(255, 0, 0), thickness=2)

# Show the result
cv2.imshow("Line Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
