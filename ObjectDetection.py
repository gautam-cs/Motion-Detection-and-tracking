import os
import numpy as np
import cv2


#custom rgb to grey scale conversion
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)


# def absolute_differnce(image1,image2):



def get_frames_and_background_substration(cwd):
    vidcap = cv2.VideoCapture(os.path.join(cwd,'TownCentreXVID.avi'))
    success=True
    count = 0
    # val = 10000
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC, val)
        success,image = vidcap.read()
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = rgb2gray(image)
        print('Read a new frame: ', success)
        if count==0:
            current_bitmap=image

        else:
            previous_bitmap=current_bitmap
            current_bitmap=image
            # diff=np.absolute(np.array(previous_bitmap) - np.array(current_bitmap))
            diff=cv2.absdiff(previous_bitmap, current_bitmap)
            ret, diff = cv2.threshold(diff,30, 255, cv2.THRESH_BINARY)
            #custom thresholding method
            # for i in range(1080):
            #       for j in range(1920):
            #           if diff[i,j]<30:
            #             diff[i,j]=0
            #           else:
            #             diff[i,j]=255


            cv2.imshow("image", diff)
            cv2.waitKey(1)
        # val = val + 10000
        count += 1
    cv2.destroyAllWindows()


if __name__=="__main__":
    cwd = os.getcwd()
    get_frames_and_background_substration(cwd)