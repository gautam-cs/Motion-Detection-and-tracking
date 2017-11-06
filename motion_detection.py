import argparse
import cv2
import numpy as np
import math
import imutils
from imutils.object_detection import non_max_suppression

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
rect = []
x = 100
y = 200
w = 30
h = 50
thresh = 100
centre = []
img = np.zeros(5)


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
            list = []
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 2)
            list.append(ix)
            list.append(iy)
            list.append(x)
            list.append(y)
            rect.append(list)
            print(list)

        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), 2)


def create_rect(img):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('c'):
            print(rect)
            break
    for j in rect:
        centre.append([(j[0] + j[2]) / 2, (j[1] + j[3]) / 2])

        # print(centre)


def calculate_distance(x, y, w, h):
    min_dist = 700000
    count = []
    point = []
    # print(x)
    point.append(x + w / 2)
    point.append(y + h / 2)
    for j in centre:
        dist = math.sqrt(int((point[0]) - j[0]) ** 2 + int(point[1] - j[1]) ** 2)
        if (dist < min_dist):
            min_dist = dist
            count = centre.index(j) + 1
            # print(count)
    print("Distance with nearest door: " + str(min_dist))
    if (min_dist < thresh):
        print("Human is near to door " + str(count))
        # print(count)


def detect_people(frame):
    """
    detect humans using HOG descriptor
    Args: frame:
    Returns: processed frame
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(frame, winStride=(16, 16), padding=(16, 16), scale=1.06)
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    temp = 0
    for (x, y, w, h) in rects:
        temp = 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        calculate_distance(x, y, w, h)
    return frame, temp, rects


# custom rgb to grey scale conversion
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)


def background_substraction(previous_frame, current_frame, min_area):
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
    temp = 0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > args["min_area"]:
            temp = 1

    return temp


if __name__ == "__main__":
    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())
    camera = cv2.VideoCapture(args["video"])
    (grabbed, frame) = camera.read()
    # initialise previous frame
    frame = imutils.resize(frame, width=900)
    cv2.setMouseCallback('frame', draw_circle)

    gray = rgb2gray(frame)
    min_area = (3000 / 800) * frame.shape[1]
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    previous_frame = gray
    flag = 1
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=900)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = rgb2gray(frame)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        current_frame = gray

        temp = background_substraction(previous_frame, current_frame, min_area)
        text = "No movement"
        text1 = "No human is there"
        if temp == 1:
            text = "Movement is detected"
            # ("someone is there")
        previous_frame = current_frame
        frame_detected, temp, rects = detect_people(frame)
        if temp == 1:
            text1 = "Human is detected"
        cv2.putText(frame_detected, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame_detected, "{}".format(text1), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("detect humans", frame_detected)
        if flag == 1:
            create_rect(frame_detected)
            flag = 0
        # calculate_distance(rects)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

