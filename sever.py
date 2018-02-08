# coding=utf-8
import os
from flask import Flask, request, render_template,jsonify
from werkzeug import secure_filename
import time
from recognizition_without_align import detect, addface
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG','jpeg','JPEG'])
#admitted forms

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#limit the upload files


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
        facecompre_name, facecompre_distance, face_flag = detect(reg_face)
        if facecompre_name == None:
            return "There is no face in the picture."
        if len(facecompre_name) == 1:
            if face_flag[0] == 0:
                return jsonify({"We don't know about you.": str(facecompre_distance[0]), "But you are most likely:": facecompre_name[0]})
            return jsonify({'The man(woman) in this image is': facecompre_name[0], 'the difference is': str(facecompre_distance[0])})
        name_list = []
        difference_list = []
        for i, name in enumerate(facecompre_name):
            if face_flag[i] == 0:

                name_list.append(u'不认识,但是你可能是'+name)
                difference_list.append([str(facecompre_distance[i])])
            else:
                difference_list.append(str(facecompre_distance[i]))
                name_list.append(name)

        return jsonify({'The people in this image are': name_list, 'The differences are': difference_list})

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)