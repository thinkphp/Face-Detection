#
#   Author   :	Adrian Statescu Dumitru
#	WebPage  :	http://adrianstatescu.ch
#   Description: Face Detection with OpenCV and Deep Learning.
#	Copyright:	Free to use and distribute as long as this note is kept.
#	Date     :	5:54 AM Tuesday, Nov 28, 2023.
#
import cv2
# Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('triatlon.png')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (123, 345, 0), 2)

# Display the output
cv2.imshow('img', img)

cv2.waitKey()

cv2.imwrite('facedetection.png', img)

# Close all windows
cv2.destroyAllWindows()
