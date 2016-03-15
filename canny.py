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

def nothing(*arg):
        pass

# cv2.namedWindow('Settings', cv2.WINDOW_AUTOSIZE)
# cv2.createTrackbar('thrs1', 'Settings', 2000, 5000, nothing)
# cv2.createTrackbar('thrs2', 'Settings', 4000, 5000, nothing)

# while True:
img = cv2.imread('img.jpg',0) # convert to grayscale on read
# thrs1 = cv2.getTrackbarPos('thrs1', 'Settings')
# thrs2 = cv2.getTrackbarPos('thrs2', 'Settings')
# edge = cv2.Canny(img, thrs1, thrs2, apertureSize=5)
# for comparison
# cv2.namedWindow('edge img')
# cv2.imshow('edge img', edge)

# cv2.namedWindow('auto canny img')
cv2.imshow('auto canny img', auto_canny(img))

img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
count = 0
for c in contours:
  print c[count][count]
  # if len(c) == 4 and count < len(contours):
  #   cv2.rectangle(img, c[count][count], c[count+1][count+1], color='r')
  # count = count + 1

# exit mode
cv2.waitKey(0)
cv2.destroyAllWindows()
