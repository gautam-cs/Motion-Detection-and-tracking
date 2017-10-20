import os
import numpy as np
import cv2
cwd=os.getcwd()

cap = cv2.VideoCapture(os.path.join(cwd,'town.avi'))

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# current_bitmap = cv2.imread("frame0.jpg", 0)
# img_count=1
# for i in range(1,2):
#     previous_bitmap=current_bitmap
#     fr="frame"+str(img_count)+".jpg"
#     print(fr)
#     current_bitmap = cv2.imread(fr,0)
#     cv2.imshow("gray",current_bitmap)
#     # diff=current_bitmap-previous_bitmap
#     diff=cv2.absdiff(current_bitmap,previous_bitmap)
#     for i in range(1080):
#         for j in range(1920):
#             if diff[i,j]<0:
#                 diff[i,j]=0
#             #else:
#                 #diff[i,j]=255
#
#     # print(len(current_bitmap))
#     # print(len(current_bitmap[0]))
#     img_count+=1


# print(bitmap)
# cv2.imshow("frame",bitmap)
# cv2.waitKey()
# cv2.destroyAllWindows()