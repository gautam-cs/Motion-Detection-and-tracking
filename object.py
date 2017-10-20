import os
import numpy as np
import cv2

def get_frames_and_background_substration(cwd):
    vidcap = cv2.VideoCapture(os.path.join(cwd,'TownCentreXVID.avi'))
    success=True
    count = 0
    val = 10000
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
            diff = cv2.absdiff(current_bitmap, previous_bitmap)
            cv2.imshow("current_bitmap", diff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # val = val + 10000
        count += 1




if __name__=="__main__":
    cwd = os.getcwd()
    get_frames_and_background_substration(cwd)