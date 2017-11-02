import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

#custom rgb to grey scale conversion
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)
 
# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 

camera = cv2.VideoCapture(args["video"])
 
# initialise previous frame
previous_frame = None


while True:

    (grabbed, frame) = camera.read()
 
    if not grabbed:
        break
 
    # resize the frame, convert it to grayscale, and blur it
    #frame = imutils.resize(frame, width=900)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=rgb2gray(frame)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    current_frame=gray

    # if the previous frame is None, initialize it
    if previous_frame is None:
        previous_frame = gray
        continue

    # compute the absolute difference between the current frame and previous frame
    frameDelta = cv2.absdiff(previous_frame, current_frame)
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    
    #custom thresholding method
            # for i in range(1080):
            #       for j in range(1920):
            #           if frameDelta[i,j]<30:
            #             frameDelta[i,j]=0
            #           else:
            #             frameDelta[i,j]=255
            
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    previous_frame=gray
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
 
camera.release()
cv2.destroyAllWindows()