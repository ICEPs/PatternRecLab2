import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)
 
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

im = cv2.imread('img.jpg') # convert to grayscale on read
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('test', imgray)
edge = auto_canny(imgray, sigma=1.0)
img, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
size = len(contours)
g = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x in range(0, size):
  c = contours[x]
  x,y,w,h = cv2.boundingRect(c)
  cv2.rectangle(g, (x,y), (x+w,y+h), (0,0,255))
cv2.imshow('image', g)

# exit mode
ch = cv2.waitKey(0)
cv2.destroyAllWindows()
