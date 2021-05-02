import cv2
import numpy as np
import cvlib as cv



# open webcam
webcam = cv2.VideoCapture('dataset_3.mp4')

counter = 0

while webcam.isOpened():
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)

    frame = cv2.resize(frame, (150, 150))


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        counter += 1
        label, filename = "Frame captured  - " + str(counter), 'dataset/dip/frame_'+str(counter) + '.png'
        
        cv2.imwrite(filename, frame)
        

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()


