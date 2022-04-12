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

"""
def group_similar (data, itterations, axis):
    # sorts array by theta
    data = data[data[:, axis].argsort()]
    print(data)
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
            i += 2
        data = np.delete(lines, (rows_to_delete), axis=0)
        rows_to_delete = []
    return data
"""
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

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

lines = group_lines(lines, 40)
print(lines)
segmented = segment_by_angle_kmeans(lines)

print(segmented)

"""
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
"""