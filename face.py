import cv2
faceCasade = cv2.CascadeClassifier("face.xml")

video_capture = cv2.VideoCapture(0)

while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    faces = faceCasade.detectMultiScale(gray,1.3,3)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xff == ord('x'):
        break

video_capture.release()
cv2.destroyAllWindows()