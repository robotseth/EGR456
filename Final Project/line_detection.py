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

# formats lines as an array of rho theta pairs
tmp = np.empty(shape=(1,2))
for line in range(len(lines)):
    tmp = np.vstack((tmp,lines[line][0]))
lines = tmp

rho_threshold = 50
theta_threshold = 5


def group_similar (data, itterations, axis):
    # sorts array by theta
    data = data[data[:, axis].argsort()]
    rows = 0
    columns = 0
    rows_to_delete = []
    #tmp_array = np.empty(shape=(1,2))
    for j in range(itterations):
        rows, columns = data.shape
        for i in range(rows - 1):
            if abs(data[i][0] - data[i+1][0]) <= rho_threshold and abs(data[i][1] - data[i+1][1]) <= theta_threshold:
                tmp_array = np.vstack((data[i], data[i+1]))
                data[i] = tmp_array.mean(axis=0)
                rows_to_delete.append(i)
        data = np.delete(lines, (rows_to_delete), axis=0)
        rows_to_delete = []
    return data

lines = group_similar(lines,5,0)
lines = group_similar(lines,5,1)
print(lines)

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
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# group lines
# average group

# Show the result
cv2.imshow("Line Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()