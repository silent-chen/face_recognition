import argparse
import os
import cv2
import numpy as np
import shutil
from collections import namedtuple
# from rcnn.logger import logger
# from timer import Timer
import datetime
import matplotlib.pyplot as plt
import sys
import json
from tempfile import TemporaryFile
import base64
import face_recognition
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image
import argparse
import imutils
import dlib
import cv2

count = 0

age_labels = ['happy 0', 'unhappy 1']
gender_labels = ['man', 'woman']

zhangyue_image = face_recognition.load_image_file("1.jpg")
xiaoming_image = face_recognition.load_image_file("00.jpg")

zhangyue_face_encoding = face_recognition.face_encodings(zhangyue_image)[0]
xiaoming_face_encoding = face_recognition.face_encodings(xiaoming_image)[0]

known_faces = [
    zhangyue_face_encoding,
    xiaoming_face_encoding
]

face_name = ['zhangsan','lisi']

Batch = namedtuple('Batch', ['data'])


def get_image(crop_image, show=False):
    # convert into format (batch, RGB, width, height)
    # TODO add the image to the center of the image.
    crop_image = cv2.imread(crop_image.encode('gbk'))
    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

    img = cv2.resize(crop_image, (256, 256))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    im = img[np.newaxis, :]
    return im

def face_predict(crop_image):
    im = get_image(crop_image, show=False)
    # compute the predict probabilities
    age_prob_temp = np.zeros((4, 2))
    gender_prob_temp = np.zeros((4, 2))

    num = 0
    for i in {8, 24}:
        for j in {8, 24}:
            # age
            mod.forward(Batch([mx.nd.array(im[:,:,j:j+224,i:i+224])]))
            age_prob_temp[num] = mod.get_outputs()[0].asnumpy()
            age_prob_temp[num] = np.squeeze(age_prob_temp[num])
            # gender
            mod1.forward(Batch([mx.nd.array(im[:, :, j:j + 224, i:i + 224])]))
            gender_prob_temp[num] = mod.get_outputs()[0].asnumpy()
            gender_prob_temp[num] = np.squeeze(gender_prob_temp)
            num += 1

    # age result
    age_prob = np.mean(age_prob_temp, axis=0)
    age = np.argsort(age_prob)[::-1]
    age_cur_pro = age_prob[age[0]]
    age_predict_name = age_labels[age[0]].split(' ')[0]
    # gender result
    gender_prob = np.mean(gender_prob_temp, axis=0)
    gender = np.argsort(gender_prob)[::-1]
    gender_cur_pro = age_prob[gender[0]]
    gender_predict_name = gender_labels[gender[0]].split(' ')[0]
    return age_predict_name, age_cur_pro, gender_predict_name, gender_cur_pro

def detectFaces(image_name):
    assert os.path.exists(image_name), image_name + ' not found'
    # print datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    image = face_recognition.load_image_file(image_name)

    face_locations = face_recognition.face_locations(image,model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    cnt = 0

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        cv2.imwrite(str(cnt).zfill(2)+'.jpg', face_image)
        cnt = cnt + 1
        # pil_image = Image.fromarray(face_image)
        # pil_image.show()
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    boxes_this_image = []
    for (x, y, width, height) in faces:
        boxes_this_image.append((x, y, x+width, y+height))
    return boxes_this_image

def align_faces(image_name, shapepredictor):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shapepredictor)
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    # load the input image, resize it, and convert it to grayscale
    # The Chinese name may occur some error.
    image = cv2.imread(image_name.encode('gbk'))
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    rects = detector(gray, 2)

    if len(rects) == 0:
        print "No face is deteced"

    cnt = 0

    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)
        facesavename = str(cnt).zfill(5)+'.jpg'
        cv2.imwrite(facesavename, faceAligned)
        
        cnt = cnt + 1

    return facesavename

def face_validation(facesavename):
    print facesavename
    unknown_image = face_recognition.load_image_file(facesavename)
    # Get the face encodings for each face in each image file
    # Since there could be more than one face in each image, it returns a list of encodings.
    # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
    
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
    facecompre_name = [face_name[i] for i in range(len(results)) if results[i] == True]

    print("Is the unknown face a picture of Zhangyue? {}".format(results[0]))
    print("Is the unknown face a picture of Xiaoming? {}".format(results[1]))
    print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

    if not True in results:
        face_name.append(unknown_face_encoding)
        print('Please input your name')
    return facecompre_name

def detect(url, shapepredictor):
    print('detect start=%s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # Detect all object classes and regress object bounds  
    print("file=" + url)
    # image_name = url.split(';')
    arr = []
    item = []
    # Detect the faces in one image and align faces.
    facesavename = align_faces(url, shapepredictor)

    # face vaildation
    facecompre_name = face_validation(facesavename)

    # face recognition
    age_predict_name, age_cur_pro, gender_predict_name, gender_cur_pro = face_predict(facesavename)

    if age_cur_pro >= 0.6:
        item.append({'age': age_predict_name, 'age_pro': age_cur_pro})
    if gender_cur_pro >= 0.7:
        item.append({'gender': gender_predict_name, 'gender_pro': gender_cur_pro})
    arr.append({'file': url, 'faces': item})
    print arr
    return arr

# load cell model
def load_agemodel(args):
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.gpu)
    global mod
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

# load nucleus model
def load_gendermodel(args):
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.classify, args.classiter)
    global mod1
    mod1 = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod1.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
             label_shapes=mod1._label_shapes)
    mod1.set_params(arg_params, aux_params, allow_missing=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Face_JOY network')
    # parser.add_argument('--sayname', help='class model', default=40, type=int)
    parser.add_argument('--prefix', help='saved model prefix', default='model/e2e', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', default=30, type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=1, type=int)
    parser.add_argument('--imagefile', help='input image', default='4.jpg', type=str)
    parser.add_argument('--shapepredictor', help='input shape-predictor', default='face-alignment/shape_predictor_68_face_landmarks.dat', type=str)
    parser.add_argument('--classify', help='class model', default='model/resnet-18', type=str)
    parser.add_argument('--classiter', help='class model', default=40, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    load_agemodel(args)
    load_gendermodel(args)
    detect(args.imagefile, args.shapepredictor)

if __name__ == '__main__':
    main()
