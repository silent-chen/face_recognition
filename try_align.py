import sys

import dlib
import cv2
import numpy as np


if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./face_alignment.py shape_predictor_5_face_landmarks.dat ../examples/faces/bald_guys.jpg\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()

predictor_path = sys.argv[1]
face_file_path = sys.argv[2]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using OpenCV
bgr_img = cv2.imread(face_file_path)
if bgr_img is None:
    print("Sorry, we could not load '{}' as an image".format(face_file_path))
    exit()

# Convert to RGB since dlib uses RGB images
img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)

num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

# Find the 5 face landmarks we need to do the alignment.
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

# Get the aligned face images
# Optionally:
# images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
images = dlib.get_face_chips(img, faces, size=320)
for image in images:
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image',cv_bgr_img)
    cv2.waitKey(0)

# It is also possible to get a single chip
image = dlib.get_face_chip(img, faces[0])
cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow('image',cv_bgr_img)
cv2.waitKey(0)

cv2.destroyAllWindows()