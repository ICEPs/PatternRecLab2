import cv2
import numpy as np
import csv

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)

  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

def sobelMap(img):
    sobelx = cv2.Sobel(img, cv2.CV_16S,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_16S,0,1,ksize=3)
    absGradX = cv2.convertScaleAbs(sobelx)
    absGradY = cv2.convertScaleAbs(sobely)
    sobelMat = cv2.addWeighted(absGradX, 0.5, absGradY, 0.5, 0)

    return sobelMat

def buildNgMat(img):
    resizedImg = cv2.resize(img, (8,8))
    return resizedImg

def getNGFeatureVector(img):
    # img must be grayscale
    features = []
    for i in range(0, 8):
        for j in range(0,8):
            features.append(img[i][j])
    return features


cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

im = cv2.imread('fig2.png',0) # convert to grayscale on read, where contours will be derived from
g = cv2.imread('fig2.png',3) # where the rects are to be drawn
to_be_cropped = cv2.imread('fig2.png',0) # image for cropping

edge = auto_canny(im, sigma=20.0)
img, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
size = len(contours)
# cropped_num = 0;
for a in range(0, size):
    c = contours[a]
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(g, (x,y), (x+w,y+h), (0,0,255))
    if(h > 10 and w > 10):
        cropped_g = to_be_cropped[y:y+h, x:x+w]
        newimg = buildNgMat(sobelMap(buildNgMat(cropped_g)));
        features = [];
        features = getNGFeatureVector(newimg)
        with open('csvfile.csv', 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            #data = [str(a), features]            
            spamwriter.writerow(features)
        #cv2.imwrite('images'+str(a)+".jpg", newimg)
        # cropped_num = cropped_num+1
cv2.imshow('image', g)

# exit mode
ch = cv2.waitKey(0)
cv2.destroyAllWindows()
