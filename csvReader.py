
import cv2
import numpy as np
import csv
import math


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

def mean(numbers):
    sum_everything = sum(numbers)
    return sum_everything/len(numbers)
    
def variance(result):
    return np.var(result)
    
def gaussian(mean, var, x):
    pi = 3.14
    e = 2.714567
    mul_one = 1.0/(math.sqrt(2.0*pi*var)+1)
    divi = -(math.pow((x - mean), 2.0))/((2.0*var)+1)
    mul_two = math.pow(e, divi)
    total = mul_one * mul_two
    return total
    
def gaussian_total(gaussian_list, prob):
    result = np.prod(np.array(gaussian_list))
    return result * prob
    

list = [] #initializing the list that will hold all values
with open('csvfile.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        #print row
		newRow = [float(n) for n in row] #turning values into integers
		list.append(newRow)
  
object_list = []
nonobject_list = []
sortedList = sorted(list,key=lambda l:l[64], reverse=True) #sorting the list by the 64th row, which classifies whether something is an object or not
for row in sortedList:
    if(row[64]==1):
        newRow = [float(n) for n in row] #turning values into integers
        object_list.append(newRow)
    else:
        newRow = [float(n) for n in row] #turning values into integers
        nonobject_list.append(newRow)

prob_one = float(float(len(object_list))/float(len(sortedList))) # number of objects / total number of images
print "lol"
print prob_one
prob_zero = float(float(len(nonobject_list))/float(len(sortedList))) # number of non - objects / total number of images
print "lol" 
print prob_zero

variance_list_one = [] #contains list of the var of each dimens/x with row[64] == 1
means_list_one = [] #contains list of the mean of each dimens/x with row[64] == 1

#this loop gets the means and variance of rows identified as an object
for a in range(0, 64):
    dimens = []
    for row in object_list:
        dimens.append(float(row[a]))
    variance_list_one.append(variance(dimens))
    means_list_one.append(mean(dimens))
    
variance_list_zero = []
means_list_zero = []

#this loop gets the means and variance of rows identified as a non-object
for a in range(0, 64):
    dimens = []
    for row in nonobject_list:
        dimens.append(float(row[a]))
    variance_list_zero.append(variance(dimens))
    means_list_zero.append(mean(dimens))


im = cv2.imread('fig3.png',0) # convert to grayscale on read, where contours will be derived from
g = cv2.imread('fig3.png',3) # where the rects are to be drawn
to_be_cropped = cv2.imread('fig3.png',0) # image for cropping

edge = auto_canny(im, sigma=20.0)
img, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
size = len(contours)
# cropped_num = 0;
for a in range(0, size):
    c = contours[a]
    x,y,w,h = cv2.boundingRect(c)
    if(h > 10 and w > 10):
        cropped_g = to_be_cropped[y:y+h, x:x+w]
        newimg = buildNgMat(sobelMap(buildNgMat(cropped_g)));
        features = [];
        features = getNGFeatureVector(newimg)
        gaussian_one = [] #stores all the gaussian result of objects/1
        gaussian_zero = [] #stores all the gaussian result of non-objects/0
        for i in range(0,64):
            gaussian_zero.append(gaussian(means_list_zero[i], variance_list_zero[i], features[i]))
            gaussian_one.append(gaussian(means_list_one[i], variance_list_one[i], features[i]))
        total_gaussian_one = gaussian_total(gaussian_one, prob_one)
        total_gaussian_zero = gaussian_total(gaussian_zero, prob_zero)
        
        if(total_gaussian_one > total_gaussian_zero):
            print "If-One"
            print total_gaussian_one
            print "If-Zero"
            print total_gaussian_zero
            cv2.rectangle(g, (x,y), (x+w,y+h), (255,0,0))
        else:
            print "Else-One"
            print total_gaussian_one
            print "Else-Zero"
            print total_gaussian_zero
            cv2.rectangle(g, (x,y), (x+w,y+h), (0,0,255))
            
cv2.imshow('image', g)


# exit mode
ch = cv2.waitKey(0)
cv2.destroyAllWindows()
