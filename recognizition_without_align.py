import argparse
import os
import cv2
import numpy as np
import shutil
from collections import namedtuple
import datetime
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


# zhangyue_image = face_recognition.load_image_file("1.jpg")
# xiaoming_image = face_recognition.load_image_file("00.jpg")
#
# zhangyue_face_encoding = face_recognition.face_encodings(zhangyue_image)[0]
# xiaoming_face_encoding = face_recognition.face_encodings(xiaoming_image)[0]
#
# known_faces = [
#     zhangyue_face_encoding,
#     xiaoming_face_encoding
# ]
#
# face_name = ['zhangsan','lisi']
known_faces = []
face_name = []
face_json = []
# I didn't find any usage of detectFaces in the project
# def detectFaces(image_name):
#     assert os.path.exists(image_name), image_name + ' not found'
#     # print datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#
#     image = face_recognition.load_image_file(image_name)
#
#     face_locations = face_recognition.face_locations(image,model="cnn")
#
#     print("I found {} face(s) in this photograph.".format(len(face_locations)))
#
#     cnt = 0
#
#     for face_location in face_locations:
#
#         # Print the location of each face in this image
#         top, right, bottom, left = face_location
#         print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
#         # You can access the actual face itself like this:
#         face_image = image[top:bottom, left:right]
#         cv2.imwrite(str(cnt).zfill(2)+'.jpg', face_image)
#         cnt = cnt + 1
#         # pil_image = Image.fromarray(face_image)
#         # pil_image.show()
#     faces = face_cascade.detectMultiScale(gray, 1.2, 5)
#     boxes_this_image = []
#     for (x, y, width, height) in faces:
#         boxes_this_image.append((x, y, x+width, y+height))
#     return boxes_this_image

# def align_faces(image_name, shapepredictor = sys.path[0]+'/models/shape_predictor_68_face_landmarks.dat'):
#     # the facial landmark predictor and the face aligner
#     #     assert os.path.exists(image_name), image_name + ' not found'
#     #     # print datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#     #
#     image = cv2.imread(image_name)
#     predictor = dlib.shape_predictor(shapepredictor)
#     fa = FaceAligner(predictor, desiredFaceWidth=256)
#     # cnn_face_detector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
#     # dets = cnn_face_detector(image, 1)
#     # rects = dlib.rectangles()
#     # for i, d in enumerate(dets):
#     #     rects.append(d.rect)
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     detector = dlib.get_frontal_face_detector()
#     # load the input image, resize it, and convert it to grayscale
#     # The Chinese name may occur some error.
#     image = imutils.resize(image, width=800)
#     image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # show the original input image and detect faces in the grayscale
#     rects = detector(gray, 2)
#
#     if(len(rects) == 0):
#         cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.path[0]+"/models/mmod_human_face_detector.dat")
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         dets = cnn_face_detector(gray, 1)
#         rects = dlib.rectangles()
#         for i, d in enumerate(dets):
#             rects.append(d.rect)
#     # rects = face_recognition.face_locations(image,model="cnn")
#
#     # print("I found {} face(s) in this photograph.".format(len(face_locations)))
#     # win.clear_overlay()
#     # win.set_image(image_RGB)
#     # win.add_overlay(rects)
#     # dlib.hit_enter_to_continue()
#
#     if len(rects) == 0:
#         print ("No face is deteced")
#
#     else:
#         print ("There is(are) {} face(s) in the picture".format(len(rects)))
#     cnt = 0
#     facesavename=[]
#     for rect in rects:
#         # extract the ROI of the *original* face, then align the face
#         # using facial landmarks
#         #(x, y, w, h) = rect_to_bb(rect)
#         #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
#         faceAligned = fa.align(image, gray, rect)
#         rootdir = sys.path[0]+'/aligned_face'
#         list = os.listdir(rootdir)
#         facesavename_tmp = sys.path[0]+'/aligned_face/' + str(len(list)).zfill(5)+'.jpg'
#         facesavename.append(facesavename_tmp)
#         cv2.imwrite(facesavename_tmp, faceAligned)
#         cnt = cnt + 1
#
#     return facesavename

def face_validation(facesavename_tmp,shapepredictor):

    print facesavename_tmp
    unknown_image = face_recognition.load_image_file(facesavename_tmp)
    detector = dlib.get_frontal_face_detector()
    # load the input image, resize it, and convert it to grayscale
    # The Chinese name may occur some error.
    #    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    rects = detector(gray, 2)
    if len(rects) == 0:
        cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.path[0]+"/models/mmod_human_face_detector.dat")
        gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
        dets = cnn_face_detector(gray, 1)
        rects = dlib.rectangles()
        for i, d in enumerate(dets):
            rects.append(d.rect)
    # rects = face_recognition.face_locations(image,model="cnn")

    # print("I found {} face(s) in this photograph.".format(len(face_locations)))
    if len(rects) == 0:
        print ("No face is deteced")
        facecompre_name = None
        facecompre_distance = None
        facesavename = None
        face_flag = None
        return facecompre_name, facecompre_distance, facesavename, face_flag
    else:
        print ("There is(are) {} face(s) in the picture".format(len(rects)))
    # Get the face encodings for each face in each image file
    # Since there could be more than one face in each image, it returns a list of encodings.
    # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
    # face_recognition_model = face_recognition_models.face_recognition_model_location()
    # face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
    # unknown_face_encoding = np.array(face_encoder.compute_face_descriptor(unknown_image, rect, 1))
    facerec = dlib.face_recognition_model_v1(sys.path[0]+"/models/dlib_face_recognition_resnet_model_v1.dat")
    predictor = dlib.shape_predictor(shapepredictor)
    facecompre_name = []
    facecompre_distance = []
    facesavename = []
    face_flag = []
    rootdir = sys.path[0] + '/added_extracted_face'
    list = os.listdir(rootdir)
    cnt = 0
    for rect in rects:
        facesavename_tmp = sys.path[0] + '/added_extracted_face/' + str(len(list)+cnt).zfill(5) + '.jpg'
        cnt = cnt+1
        cv2.imwrite(facesavename_tmp, cv2.cvtColor(unknown_image[rect.top():rect.bottom(), rect.left():rect.right()], cv2.COLOR_BGR2RGB))
        facesavename.append(facesavename_tmp)
        shape = predictor(unknown_image, rect)
        # win = dlib.image_window()
        # win.clear_overlay()
        # win.set_image(unknown_image)
        # dlib.hit_enter_to_continue()
        # face_locations =  face_recognition.face_locations(unknown_image)
        #     face_locations = face_recognition.face_locations(unknown_image, model="cnn")
        # if face_locations == None:
        #     return None
        unknown_face_encoding = np.array(facerec.compute_face_descriptor(unknown_image, shape))
        # unknown_face_encoding = face_recognition.face_encodings(unknown_image, face_locations)[0]

        # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
        results = face_recognition.compare_faces(known_faces, unknown_face_encoding, 0.44)
        min_position = -1
        min_distance = 1000
        for i in range(len(results)):
            now_distance = face_recognition.face_distance([known_faces[i]], unknown_face_encoding)
            if (min_distance > now_distance):
                min_distance = now_distance
                min_position = i
        print min_distance
        if not True in results:
            print("We don't know who are you.")
            face_flag.append(0)
            facecompre_name.append(face_name[min_position])
            facecompre_distance.append(min_distance)
        else:
            face_flag.append(1)
            facecompre_name.append(face_name[min_position])
            facecompre_distance.append(min_distance)
            print(face_name[min_position].encode('utf-8'))
    return facecompre_name, facecompre_distance, facesavename,face_flag

def detect(url, shapepredictor = sys.path[0]+'/models/shape_predictor_68_face_landmarks.dat'):
    _init_()
    print('detect start=%s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # Detect all object classes and regress object bounds
    print('file=' + url)
    # image_name = url.split(';')
    # arr = []
    # item = []
    # Detect the faces in one image and align faces.
    facecompre_name, facecompre_distance, facesavenamem, face_flag = face_validation(url, shapepredictor)
    # facecompre_name = []
    # # face vaildation
    # for i in range(len(facesavename)):
    #     facecompre_name.append(face_validation(facesavename[i]))
    return facecompre_name, facecompre_distance, face_flag
    # arr.append({'file': url, 'faces': item})
    # print arr
    # return arr


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Face_JOY network')
    # parser.add_argument('--sayname', help='class model', default=40, type=int)
    parser.add_argument('--prefix', help='saved model prefix', default='model/age', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', default=10, type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=1, type=int)
    parser.add_argument('--imagefile', help='input image', default=sys.path[0]+'/141637734.jpg', type=str)
    parser.add_argument('--shapepredictor', help='input shape-predictor', default=sys.path[0]+'/models/shape_predictor_68_face_landmarks.dat', type=str)
    args = parser.parse_args()
    return args

def init_csv2list(fileName="", dataList=[], splitsymbol=","):
    with open(fileName, "r") as csvFile:
        dataLine = csvFile.readline().strip("\n")
        while dataLine != "":
            tmpList = dataLine.split(splitsymbol)
            dataList.append(tmpList)
            dataLine = csvFile.readline().strip("\n")
        csvFile.close()

def _init_(fileName = ''):
    global known_faces
    global face_name
    known_faces = []
    face_name = []
    rootdir = sys.path[0]
    rootdir += '/face_code'
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        filepath = os.path.join(rootdir, str(i+1).zfill(5)+'.npy')
        if os.path.isfile(filepath):
            new_face = np.load(filepath)
            known_faces.append(new_face)
    rootdir = sys.path[0]
    rootdir += '/face_inf'
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        filepath = os.path.join(rootdir, str(i+1).zfill(5)+'.json')
        if os.path.isfile(filepath):
            with open(filepath) as json_obj:
                if os.path.getsize(filepath) == 0:
                    print('the face_database is empty')
                    json_obj.close
                    return 0
                faces_inf = json.load(json_obj)
                face_name.append(faces_inf.get("name"))
                json_obj.close()

# def save_face_inf(fileName = ''):
#     with open(fileName, 'w') as json_obj:
#         # face_dict_new = dict(zip(face_name, known_faces))
#         # j = json.dumps(face_dict_new)
#
#         json_obj.close()

# def save_lis2csv(filename="", datalist=[]):
#     with open(filename, "wb") as csvfile:
#         csvWriter = csv.writer(csvfile)
#         for data in datalist:
#             csvWriter.writerow(data)
#        csvfile.close
def addface(filepath,unknown_face_name, force=0):
    _init_()
    # image_name = url.split(';')
    # arr = []
    # item = []
    # Detect the faces in one image and align faces.
    print filepath
    unknown_image = face_recognition.load_image_file(filepath)
    detector = dlib.get_frontal_face_detector()
    # load the input image, resize it, and convert it to grayscale
    # The Chinese name may occur some error.
    #    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    rects = detector(gray, 2)

    if len(rects) == 0:
        cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.path[0] + "/models/mmod_human_face_detector.dat")
        gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
        dets = cnn_face_detector(gray, 1)
        rects = dlib.rectangles()
        for i, d in enumerate(dets):
            rects.append(d.rect)
    # rects = face_recognition.face_locations(image,model="cnn")

    # print("I found {} face(s) in this photograph.".format(len(face_locations)))

    if len(rects) == 0:
        print ("No face is deteced")
        return None
    else:
        print ("There is(are) {} face(s) in the picture".format(len(rects)))
    if len(rects) >= 2:
        print ("There two more faces in the picture".format(len(rects)))
        return None
    facerec = dlib.face_recognition_model_v1(sys.path[0] + "/models/dlib_face_recognition_resnet_model_v1.dat")
    predictor = dlib.shape_predictor(sys.path[0]+'/models/shape_predictor_68_face_landmarks.dat')
    rootdir = sys.path[0] + '/added_extracted_face'
    list = os.listdir(rootdir)
    cnt = 0
    for rect in rects:
        facesavename_tmp = sys.path[0] + '/added_extracted_face/' + str(len(list)+cnt).zfill(5) + '.jpg'
        cnt = cnt+1
        cv2.imwrite(facesavename_tmp, cv2.cvtColor(unknown_image[rect.top():rect.bottom(), rect.left():rect.right()], cv2.COLOR_BGR2RGB))
        facecompre_distance = []
        shape = predictor(unknown_image, rect)
        # win = dlib.image_window()
        # win.clear_overlay()
        # win.set_image(unknown_image)
        # dlib.hit_enter_to_continue()
        # face_locations =  face_recognition.face_locations(unknown_image)
        #     face_locations = face_recognition.face_locations(unknown_image, model="cnn")
        # if face_locations == None:
        #     return None
        unknown_face_encoding = np.array(facerec.compute_face_descriptor(unknown_image, shape))
        # unknown_face_encoding = face_recognition.face_encodings(unknown_image, face_locations)[0]

        # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
        results = face_recognition.compare_faces(known_faces, unknown_face_encoding, 0.44)
        facecompre_distance_tmp = [face_recognition.face_distance([known_faces[i]], unknown_face_encoding) for i in range(len(results)) if results[i] == True]
        if (True in results) and (force == 0):
            facecompre_distance.append(facecompre_distance_tmp)
            return facecompre_distance
        else:
            print("now known_faces: {}".format(len(known_faces)))
            np.save(sys.path[0] + '/face_code/' + str(len(known_faces) + 1).zfill(5) + '.npy', unknown_face_encoding)
            known_faces.append(unknown_face_encoding)
            face_name.append(unknown_face_name)
            with open(sys.path[0] + '/face_inf/' + str(len(known_faces)).zfill(5) + '.json', 'w') as json_obj:
                json.dump({"name": unknown_face_name.encode('utf-8')}, json_obj)
                json_obj.close()
            return 1

def main():
    # init_csv2list('face_name.csv', face_name)
    # init_csv2list('known_faces.csv', known_faces)
    args = parse_args()
    detect(args.imagefile, args.shapepredictor)
    # save_face_inf('face_inf.json')
    # save_lis2csv('face_name.csv',face_name)
    # save_lis2csv('known_faces.csv', known_faces)

if __name__ == '__main__':
    main()
