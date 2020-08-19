import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
webcam = cv2.VideoCapture(0)

while True:
    successful_read, frame = webcam.read()
    if not successful_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale)
    

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)

        the_face = frame[x:x+w, y:y+h] 
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smile = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
        for(xs, ys, ws, hs) in smile:
            image = cv2.rectangle(the_face, (xs , ys), (xs + ws , ys + hs ), (5, 5, 200), 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'Smiling',(0,100), font, 1, (50, 50, 200), 3)

    cv2.imshow('SmileIt         (press q to close.)', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()