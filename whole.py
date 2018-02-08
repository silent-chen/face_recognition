# coding=utf-8
import argparse
import os
import cv2
import mxnet as mx
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
from flask import Flask, request, render_template,jsonify
from werkzeug import secure_filename
import time
# from recognizition import detect, addface
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG','jpeg','JPEG'])
#admitted forms

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

count = 0

# age_labels = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_labels = ['Male','Female']
emotion_labels = ['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

age_labels = range(0,100)

known_faces = []
face_name = []
face_json = []

Batch = namedtuple('Batch', ['data'])



def get_image(crop_image, show=False):
    # convert into format (batch, RGB, width, height)
    # TODO add the image to the center of the image.
    # crop_image = cv2.imread(crop_image.encode('gbk'))
    print crop_image
    print 'eeeeeeeeeeeeeeeeeeeee'
    crop_image = cv2.imread(crop_image)
    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

    img = cv2.resize(crop_image, (256, 256))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    im = img[np.newaxis, :]
    return im

def face_predict(crop_image):
    im = get_image(crop_image, show=False)
    # compute the predict probabilities
    age_prob_temp = np.zeros((4, 101))
    gender_prob_temp = np.zeros((4, 2))
    emotion_prob_temp = np.zeros((4, 7))
    num = 0
    for i in {8, 24}:
        for j in {8, 24}:
            # age
            mod.forward(Batch([mx.nd.array(im[:,:,j:j+224,i:i+224])]))
            age_prob_temp[num] = mod.get_outputs()[0].asnumpy()
            age_prob_temp[num] = np.squeeze(age_prob_temp[num])
            # gender
            mod1.forward(Batch([mx.nd.array(im[:, :, j:j + 227, i:i + 227])]))
            gender_prob_temp[num] = mod1.get_outputs()[0].asnumpy()
            gender_prob_temp[num] = np.squeeze(gender_prob_temp[num])
            # emotion
            mod2.forward(Batch([mx.nd.array(im[:, :, j:j + 224, i:i + 224])]))
            emotion_prob_temp[num] = mod2.get_outputs()[0].asnumpy()
            emotion_prob_temp[num] = np.squeeze(emotion_prob_temp[num])
            num += 1

    # age result
    age_prob = np.mean(age_prob_temp, axis=0)
    age = np.argsort(age_prob)[::-1]
    age_cur_pro = age_prob[age[0]]
    age_predict_name = age_labels[age[0]]
    # gender result
    gender_prob = np.mean(gender_prob_temp, axis=0)
    gender = np.argsort(gender_prob)[::-1]
    gender_cur_pro = gender_prob[gender[0]]
    gender_predict_name = gender_labels[gender[0]].split(' ')[0]

    # emotion result
    emotion_prob = np.mean(emotion_prob_temp, axis=0)
    emotion = np.argsort(emotion_prob)[::-1]
    emotion_cur_pro = emotion_prob[emotion[0]]
    emotion_predict_name = emotion_labels[emotion[0]].split(' ')[0]

    return age_predict_name, age_cur_pro, gender_predict_name, gender_cur_pro, emotion_predict_name, emotion_cur_pro

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

def face_validation(facesavename_tmp, shapepredictor = sys.path[0]+'/models/shape_predictor_68_face_landmarks.dat'):

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
    return facecompre_name, facecompre_distance, facesavename, face_flag

def detect(url, shapepredictor = sys.path[0]+'/models/shape_predictor_68_face_landmarks.dat'):
    _init_()
    print('detect start=%s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # Detect all object classes and regress object bounds
    print('file=' + url)
    # image_name = url.split(';')
    # arr = []
    # item = []
    # Detect the faces in one image and align faces.
    # facecompre_name, facecompre_distance = face_validation(url, shapepredictor)
    print 'ooooooooooooo'
    # facecompre_name = []
    # facecompre_distance = []
    # face vaildation
    age_predict_name = []
    age_cur_pro = []
    gender_predict_name = []
    gender_cur_pro = []
    emotion_predict_name = []
    emotion_cur_pro = []
    facecompre_name, facecompre_distance, facesavename ,face_flag = face_validation(url)
    for i in range(len(facesavename)):
        # facecompre_name.append(face_validation(facesavename[i]))
        # facecompre_distance.append([min_distance])
        # print(face_name[min_position].encode('utf-8')
        age_predict_name_tmp, age_cur_pro_tmp, gender_predict_name_tmp, gender_cur_pro_tmp, emotion_predict_name_tmp, emotion_cur_pro_tmp = face_predict(facesavename[i])
        age_predict_name.append(age_predict_name_tmp)
        age_cur_pro.append(age_cur_pro_tmp)
        gender_predict_name.append(gender_predict_name_tmp)
        gender_cur_pro.append(gender_cur_pro_tmp)
        emotion_predict_name.append(emotion_predict_name_tmp)
        emotion_cur_pro.append(emotion_cur_pro_tmp)
        print age_predict_name_tmp, age_cur_pro_tmp, gender_predict_name_tmp, gender_cur_pro_tmp
    return facecompre_name, facecompre_distance, face_flag, age_predict_name, age_cur_pro, gender_predict_name, gender_cur_pro, emotion_predict_name, emotion_cur_pro
    # arr.append({'file': url, 'faces': item})
    # print arr
    # return arr


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
                # face_name.append(faces_inf.get("name").encode('utf-8', 'ignore'))
                face_name.append(faces_inf.get("name"))
                json_obj.close()


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
        image_RGB = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
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


# load cell model
def load_agemodel(args):
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    global mod
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    print 'load age model done'

# load nucleus model
def load_gendermodel(args):
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.classify, args.classiter)
    global mod1
    mod1 = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod1.bind(for_training=False, data_shapes=[('data', (1,3,227,227))],
             label_shapes=mod1._label_shapes)
    mod1.set_params(arg_params, aux_params, allow_missing=True)
    print 'load gender model'

# load emotion model
def load_emotionmodel(args):
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.emotionmodel, args.emotionepoch)
    global mod2
    mod2 = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod2.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
             label_shapes=mod2._label_shapes)
    mod2.set_params(arg_params, aux_params, allow_missing=True)
    print 'loaded the emotion model'

def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Face_JOY network')
    # parser.add_argument('--sayname', help='class model', default=40, type=int)
    parser.add_argument('--prefix', help='saved model prefix', default='model/new_age', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', default=10, type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=2, type=int)
    parser.add_argument('--imagefile', help='input image', default='4.jpg', type=str)
    parser.add_argument('--shapepredictor', help='input shape-predictor', default='face-alignment/shape_predictor_68_face_landmarks.dat', type=str)
    parser.add_argument('--classify', help='class model', default='model/gender', type=str)
    parser.add_argument('--classiter', help='class model', default=10, type=int)
    parser.add_argument('--emotionmodel', help='class model', default='model/emotion', type=str)
    parser.add_argument('--emotionepoch', help='class model', default=10, type=int)
    args = parser.parse_args()
    return args

#judge the files is admited
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#the upload page
@app.route('/recognizition')
def upload_re():
    return render_template('recognizition.html')

@app.route('/upload')
def upload_up():
    return render_template('upload.html')

# upload picture and return the number
@app.route('/api/recognizition', methods=['POST'], strict_slashes=False)
def api_recognizition():
    f = request.files['file']  # use name 'file' to get the file
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        reg_face = os.path.join(app.config['UPLOAD_FOLDER']+'/reg_face', filename)
        f.save(reg_face)
        facecompre_name, facecompre_distance, face_flag, age_predict_name, age_cur_pro, gender_predict_name, gender_cur_pro,emotion_predict_name, emotion_cur_pro = detect(reg_face)
        if facecompre_name == None:
            return "There is no face in the picture."
        if len(facecompre_name) == 1:
            if face_flag[0] == 0:
                return jsonify({"But you are most likely:": facecompre_name[0],"We don't know about you.": str(facecompre_distance[0])})
            return jsonify({'The man(woman) in this image is': facecompre_name[0], 'the difference is': str(facecompre_distance[0]), 'age': age_predict_name[0], 'age_pro': age_cur_pro[0],
                'gender': gender_predict_name[0], 'gender_pro': gender_cur_pro[0],'emotion': emotion_predict_name[0], 'emotion_pro': emotion_cur_pro[0]})
        name_list = []
        difference_list = []
        for i, name in enumerate(facecompre_name):
            if face_flag[i] == 0:

                name_list.append(u'不认识,但是你可能是'+name)
                difference_list.append([str(facecompre_distance[i])])
            else:
                difference_list.append(str(facecompre_distance[i]))
                name_list.append(name)

        return jsonify({'The people in this image are': name_list, 'The differences are': difference_list, 'age': age_predict_name, 'age_pro': age_cur_pro,
                'gender': gender_predict_name, 'gender_pro': gender_cur_pro,'emotion': emotion_predict_name, 'emotion_pro': emotion_cur_pro})

@app.route('/api/force_upload', methods=['POST'], strict_slashes=False)
def api_force_upload():
    f = request.files['file']# use name 'file' to get the file
    username = (request.form['username'])
    if(len(username) == 0):
        return "Please enter your name."
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        file_add = os.path.join(app.config['UPLOAD_FOLDER']+'/added_face', filename)
        f.save(file_add)
        result = addface(file_add, username,1)
        if result == None:
            return "There is no face or two more faces in the picture."
        else:
            return "You have successfully added your face into the faces database."

@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    f = request.files['file']# use name 'file' to get the file
    username = (request.form['username'])
    if(len(username) == 0):
        return "Please enter your name."
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        file_add = os.path.join(app.config['UPLOAD_FOLDER']+'/added_face', filename)
        f.save(file_add)
        result = addface(file_add, username)
        if result == None:
            return "There is no face or two more faces in the picture."
        elif result == 1:
            return "You have successfully added your face into the faces database."
        print(str(result[0][0]))
        return render_template('force_upload.html')

def main():
    args = parse_args()
    load_agemodel(args)
    load_gendermodel(args)
    load_emotionmodel(args)
    app.run(host='0.0.0.0', port=5000)
    detect(args.imagefile, args.shapepredictor)

if __name__ == '__main__':
    main()
