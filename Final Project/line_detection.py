import numpy as np
import cv2
from scipy import ndimage as ndi
from collections import defaultdict

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot

img = cv2.imread('test1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = ndi.filters.median_filter(gray, size=2)

cv2.imshow("SMOOTHED", smooth)
cv2.waitKey(0)

edges = smooth > 180

lines = cv2.HoughLines(edges.astype(np.uint8), .4, np.pi/180, 120)

#print(lines)

line_array = np.empty(shape=(1,2))
for i in range(len(lines)):
    line_array = np.append(line_array,lines[i],axis=0)

# define the model
model = AffinityPropagation(damping=0.9)
# fit the model
model.fit(line_array)
# assign a cluster to each example
yhat = model.predict(line_array)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    row_ix = where(yhat == cluster)
    #print(row_ix)
    rho_array = np.average(line_array[row_ix, 0], axis=0)
    theta_array = np.average(line_array[row_ix, 1], axis=0)
    pyplot.scatter(line_array[row_ix, 0], line_array[row_ix, 1])
# show the plot
pyplot.show()


avg_lines = np.hstack((np.vstack(rho_array),np.vstack(theta_array)))
print(avg_lines)

for i in range(len(lines)):
    for rho,theta in lines[i]:
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