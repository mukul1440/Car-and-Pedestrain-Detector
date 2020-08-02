import cv2
video=cv2.VideoCapture('videoplayback.mp4')
car_classifier=cv2.CascadeClassifier('cars.xml')
pedestrain_classifier=cv2.CascadeClassifier('Pedestrain_classifier.xml')
while True:
    (read_sucessfull,frame)=video.read()
    if read_sucessfull:
        gray_scaled=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
    cars=car_classifier.detectMultiScale(gray_scaled)
    person=pedestrain_classifier.detectMultiScale(gray_scaled)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    for (x,y,w,h) in person:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow('cars',frame)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
video.release()

