import cv2

img = cv2.imread("this-person-does-not-exist.jpg")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
if len(faces) > 0:
    x, y, w, h = faces[0]
    imgCropped = img[y:y + h, x:x + w]

    cv2.imshow("Image Cropped", imgCropped)
    cv2.waitKey(0)





