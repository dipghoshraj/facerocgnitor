import numpy as np
import cv2
import os
import cvlib as cv

frame = cv2.imread('img.jpg')

face, confidence = cv.detect_face(frame)

# loop through detected faces
for idx, f in enumerate(face):

    # get corner points of face rectangle        
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]


    face_crop = np.copy(frame[startY:endY,startX:endX])

cv2.imshow("frame", face_crop)

cv2.waitKey(1000)
cv2.destroyAllWindows()