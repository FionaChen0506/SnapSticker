from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import dlib
from sticker_utils import add_ear_stickers

app = Flask(__name__)


# Path for the uploads relative to the static folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = os.path.join('static', UPLOAD_FOLDER)

# Ensure the uploads directory exists
uploads_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)


# Function to detect faces
def face_detecting(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)
    return faces


@app.route("/", methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            # Save the file to the uploads directory
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)
            
            # Detect faces and get coordinates
            image = cv2.imread(file_path)
            # Convert the image to max height 600 px
            height, width = image.shape[:2]
            scale = 600 / height
            image = cv2.resize(image, (int(width * scale), 600))
            faces = face_detecting(image)
            face_data = [{'left': face.left(), 'top': face.top(), 'right': face.right(), 'bottom': face.bottom()} for face in faces]

            # Generate the web-friendly path
            web_path = os.path.join(UPLOAD_FOLDER, filename)
            return render_template("index.html", uploaded_image=web_path, face_data=face_data)

            # if 'add_stickers' in request.form and request.form['add_stickers'] == 'true':
            #     # uploaded_image_path = request.form['uploaded_image']
            #     # # print("uploaded_image_path in main:", uploaded_image_path)
            #     # full_image_path = os.path.join(app.root_path, 'static', uploaded_image_path)
            #     # # print("Full image path:", full_image_path)  
            #     # # Add ear stickers to the uploaded image
            #     # modified_image_path = add_ear_stickers(full_image_path)
                
            # return render_template("index.html", uploaded_image=web_path, face_data=face_data, modified_image_path=web_path)
    return render_template("index.html", uploaded_image=None)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

