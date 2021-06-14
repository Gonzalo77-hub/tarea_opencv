import cv2

escala = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(escala)

webcam = cv2.VideoCapture(0)


while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor( im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)
    for (x, y , w, h) in faces:
        cv2.rectangle(im, (x,y), (x+w, y+h), (255, 0, 0), 4)
        cv2.putText(im, 'Cara', (x+100,y-10), 2, 0.7,(255,0,0),2,cv2.LINE_AA)

    cv2.imshow('img', im)
    
    key = cv2.waitKey(10)
    if key == 27:
        break