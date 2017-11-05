import argparse
import cv2
import numpy as np
import math

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
rects = []
x=100
y=200
w=30
h=50
point=[]
point.append(x+w/2)
point.append(y+h/2)
thresh=200

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode


    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing == True:
    #         if mode == True:
    #             cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
    #         else:
    #             cv2.circle(img, (x, y), 5, (0, 0, 25), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            list=[]
            cv2.rectangle(img, (ix, iy), (x, y), (0, 25, 0), 2)
            list.append(ix)
            list.append(iy)
            list.append(x)
            list.append(y)
            rects.append(list)
            print(list)

        else:
            cv2.circle(img, (x, y), 5, (0, 0, 25), 2)

def detect_people(frame):
    """
    detect humans using HOG descriptor
    Args: frame:
    Returns: processed frame
    """
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    print(type(rects))
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame


# custom rgb to grey scale conversion
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

def background_substraction(camera):
    # initialise previous frame
    previous_frame = None
    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break
        # resize the frame, convert it to grayscale, and blur it
        # frame = imutils.resize(frame, width=900)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = rgb2gray(frame)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        current_frame = gray

        # if the previous frame is None, initialize it
        if previous_frame is None:
            previous_frame = gray
            continue

        # compute the absolute difference between the current frame and previous frame
        frameDelta = cv2.absdiff(previous_frame, current_frame)
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]

        # custom thresholding method
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
            if cv2.contourArea(c) > args["min_area"]:
                temp=1
                # compute the bounding box for the contour, draw it on the frame,
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        previous_frame = gray
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    return temp

if __name__=="__main__":
    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())
    camera = cv2.VideoCapture(args["video"])
    temp=background_substraction(camera)
    img = cv2.imread("abc.jpg")
    cv2.namedWindow('image')

    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('c'):
            print(rects)
            break
    centre = []
    for j in rects:
        centre.append([(j[0] + j[2]) / 2, (j[1] + j[3]) / 2])

    print(centre)
    min_dist = 700000
    count = []
    for j in centre:
        dist = math.sqrt(int((point[0]) - j[0]) ** 2 + int(point[1] - j[1]) ** 2)
        if (dist < min_dist):
            min_dist = dist
            count = j
    print(min_dist)
    if (min_dist < thresh):
        print("Point is near")
        print(count)
    else:
        print(" no door is near to object")
    cv2.destroyAllWindows()