import cv2

path = "/Users/apple/Desktop/python_lesson/asstes/IMG_2908.JPG"
img = cv2.imread(path)
# orb = cv2.ORB_create(nfeatures=1000)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# keypoints, descriptors = orb.detectAndCompute(gray, None)
# img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
path_face = "/Users/apple/Desktop/python_lesson/haarcascade_frontalface_default.txt"
face_cascade = cv2.CascadeClassifier(path_face)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
if len(faces) == 0:
    print("No face detected.")
else:
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
