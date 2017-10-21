import os
import numpy as np
import cv2

def get_frames_and_background_substration(cwd):
    vidcap = cv2.VideoCapture(os.path.join(cwd,'TownCentreXVID.avi'))
    success=True
    count = 0
    # val = 10000
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC, val)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        # cv2.imwrite(os.path.join(cwd+"/Frames/"+"frame%d.jpg") % count, image)     # save frame as JPEG file
        # fr = os.path.join(cwd + "/Frames/" + "frame" + str(count) + ".jpg")
        if count==0:
            # current_bitmap = cv2.imread(fr,0)
            current_bitmap=image

        else:
            previous_bitmap=current_bitmap
            # current_bitmap = cv2.imread(fr, 0)
            current_bitmap=image
            diff = cv2.absdiff( previous_bitmap, current_bitmap)
            # thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            # thresh = cv2.dilate(thresh, None, iterations=2)
            # im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # temp = 0
            # min_area = (3000 / 800) * current_bitmap.shape[1]
            # for c in cnts:
            #     # if the contour is too small, ignore it
            #     if cv2.contourArea(c) > min_area:
            #         temp = 1
            cv2.imshow("image", diff)
            cv2.waitKey(1)
        # val = val + 10000
        count += 1
    cv2.destroyAllWindows()

if __name__=="__main__":
    cwd = os.getcwd()
    get_frames_and_background_substration(cwd)