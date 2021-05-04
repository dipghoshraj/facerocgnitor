import cv2
import numpy as np
import cvlib as cv



# open webcam
webcam = cv2.VideoCapture(0)

counter = 0

while webcam.isOpened():
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):
        
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        face_crop = np.copy(frame[startY:endY,startX:endX])

        # if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
        # frame = cv2.resize(face_crop, (150,150))
        frame = face_crop

        counter += 1
        label, filename = "Frame captured  - " + str(counter), 'dataset/dip/frame_'+str(counter) + '.png'
        
        # cv2.imwrite(filename, frame)
        
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


