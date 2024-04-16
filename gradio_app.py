import gradio as gr
import cv2
import numpy as np
from PIL import Image
import dlib
import os
import math

from constants import *

MAX_EXPECTED_FACES=7
# get a list of faces in the image
def face_detecting(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)
    return faces

# show all the faces in rectangles in the image
def face_showing(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
    return image

# highlight the selected face in the image, using index to select the face
def face_selecting(image, faces, index):
    face = faces[index]
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
    return image

# get the landmarks of the face
def face_landmarking(image, face):
    predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
    landmarks = predictor(image, face)
    return landmarks


# Function to overlay a transparent image onto another image
def overlay_transparent(background, overlay, x, y):
    bg_height, bg_width = background.shape[:2]
    if x >= bg_width or y >= bg_height:
        return background

    h, w = overlay.shape[:2]
    if x + w > bg_width:
        w = bg_width - x
        overlay = overlay[:, :w]

    if y + h > bg_height:
        h = bg_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate([overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255], axis=2)

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img

    return background


def calculate_eye_angle(landmarks, left_eye_indices, right_eye_indices):

    # Calculate the center point of the left eye
    left_eye_center = (
        sum([landmarks.part(i).x for i in left_eye_indices]) // len(left_eye_indices),
        sum([landmarks.part(i).y for i in left_eye_indices]) // len(left_eye_indices)
    )

    # Calculate the center point of the right eye
    right_eye_center = (
        sum([landmarks.part(i).x for i in right_eye_indices]) // len(right_eye_indices),
        sum([landmarks.part(i).y for i in right_eye_indices]) // len(right_eye_indices)
    )

    # Calculate the differences in the x and y coordinates between the centers of the eyes
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]

    # Calculate the angle using the arctangent of the differences
    angle = math.degrees(math.atan2(dy, dx))

    return angle


# Function to add ear stickers
def add_ears_sticker(img_bgr, sticker_path, faces):
    ears_pil = Image.open(sticker_path)

    # Check the color mode and convert to RGBA    
    ears_rgba = ears_pil.convert('RGBA')
    
    # Convert the ears_rgba to BGRA
    r, g, b, a = ears_rgba.split()
    ears_bgra = Image.merge("RGBA", (b, g, r, a))
    

    # A copy of the original image
    img_with_stickers = img_bgr.copy()

    for face in faces:
        landmarks = face_landmarking(img_bgr, face)

        # the landmarks 68 to 80 are for the forehead
        forehead = [landmarks.part(i) for i in range(68, 81)]

        # The landmarks 36 to 41 are for the left eye, and 42 to 47 are for the right eye
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        # Calculate the center point between the eyes
        left_eye_center = ((left_eye[0].x + left_eye[3].x) // 2, (left_eye[0].y + left_eye[3].y) // 2)
        right_eye_center = ((right_eye[0].x + right_eye[3].x) // 2, (right_eye[0].y + right_eye[3].y) // 2)

        # Calculate the angle of tilt
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = math.degrees(math.atan2(dy, dx))  


        # Calculate the bounding box for the ears based on the eye landmarks
        ears_width = int(abs(forehead[0].x - forehead[-1].x) * 2.1) 
        ears_height = int(ears_width * ears_bgra.height / ears_bgra.width)

        # Resize the ears image
        resized_ears_pil = ears_bgra.resize((ears_width, ears_height))
        rotated_ears = resized_ears_pil.rotate(-angle, expand=True, resample=Image.BICUBIC)
        
        # Calculate the position for the ears
        y1 = min([point.y for point in forehead]) - int(0.7 * ears_height)
        x1 = forehead[0].x - int(0.2 * ears_width) 

        # Convert PIL image to NumPy array
        # ears_np = np.array(resized_ears_pil)
        ears_np = np.array(rotated_ears)
        
        # Overlay the ears on the image
        img_with_stickers = overlay_transparent(img_with_stickers, ears_np, x1, y1)
        
    return img_with_stickers

# Function to add hats stickers
def add_hats_sticker(img_bgr, sticker_path, faces):
    hat_pil = Image.open(sticker_path)

    # Check the color mode and convert to RGBA
    hat_rgba = hat_pil.convert('RGBA')
    
    # Convert the hat_rgba to BGRA
    r, g, b, a = hat_rgba.split()
    hat_bgra = Image.merge("RGBA", (b, g, r, a))
    
    # A copy of the original image
    img_with_stickers = img_bgr.copy()

    for face in faces:
        landmarks = face_landmarking(img_bgr, face)

        # The landmarks 36 to 41 are for the left eye, and 42 to 47 are for the right eye
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        # Calculate the center point between the eyes
        left_eye_center = ((left_eye[0].x + left_eye[3].x) // 2, (left_eye[0].y + left_eye[3].y) // 2)
        right_eye_center = ((right_eye[0].x + right_eye[3].x) // 2, (right_eye[0].y + right_eye[3].y) // 2)
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2

        # Calculate the angle of tilt
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = math.degrees(math.atan2(dy, dx))  

        # Calculate the size of the hat based on the width between the eyes
        hat_width = int(abs(left_eye[0].x - right_eye[3].x) * 1.7)
        hat_height = int(hat_width * hat_bgra.height / hat_bgra.width)

        # Resize and rotate the hat image
        resized_hat = hat_bgra.resize((hat_width, hat_height))
        rotated_hat = resized_hat.rotate(-0.8*angle, expand=True, resample=Image.BICUBIC)

        # Calculate the position for the hat
        y1 = eye_center_y - hat_height - int(0.3 * hat_height)
        # x1 = eye_center_x - (hat_width // 2)  # Centering the hat on the midpoint between the eyes
        x1 = eye_center_x - (hat_width // 2) - int(0.03 * hat_width)  # Moving the hat a bit further to the left
        # Convert PIL image to NumPy array
        hat_np = np.array(rotated_hat)

        # Overlay the hat on the image
        img_with_stickers = overlay_transparent(img_with_stickers, hat_np, x1, y1)

    return img_with_stickers

def add_headbands_sticker(img_bgr, sticker_path, faces):
    headband_pil = Image.open(sticker_path)

    # Check the color mode and convert to RGBA
    headband_rgba = headband_pil.convert('RGBA')
    
    # Convert the headband_rgba to BGRA
    r, g, b, a = headband_rgba.split()
    headband_bgra = Image.merge("RGBA", (b, g, r, a))
    
    # A copy of the original image
    img_with_stickers = img_bgr.copy()

    for face in faces:
        landmarks = face_landmarking(img_bgr, face)

        # Determine the forehead region using landmarks
        # Assuming the headband will be placed between the temples
        left_temple = landmarks.part(0)   
        right_temple = landmarks.part(16) 

        # Calculate the width of the headband based on the temples
        headband_width = int(abs(left_temple.x - right_temple.x) * 1.6)
        headband_height = int(headband_width * headband_bgra.height / headband_bgra.width)

        # Resize the headband image
        resized_headband = headband_bgra.resize((headband_width, headband_height))

        # Calculate the angle of tilt using the eyes as reference
        left_eye_indices = range(36, 42)
        right_eye_indices = range(42, 48)
        angle = calculate_eye_angle(landmarks, left_eye_indices, right_eye_indices)

        # Rotate the headband image
        rotated_headband = resized_headband.rotate(-angle, expand=True, resample=Image.BICUBIC)

        # Calculate the position for the headband
        x1 = (left_temple.x + right_temple.x) // 2 - (headband_width // 2)
        y1 = left_temple.y - (headband_height // 2) 

        # Convert PIL image to NumPy array
        headband_np = np.array(rotated_headband)

        # Overlay the headband on the image
        img_with_stickers = overlay_transparent(img_with_stickers, headband_np, x1, y1)

    return img_with_stickers


# Function to add glasses stickers
def add_glasses_sticker(img_bgr, sticker_path, faces):
    glasses_pil = Image.open(sticker_path)

    # Check the color mode and convert to RGBA    
    glasses_rgba = glasses_pil.convert('RGBA')
    
    # Convert the glasses_rgba to BGRA
    r, g, b, a = glasses_rgba.split()
    glasses_bgra = Image.merge("RGBA", (b, g, r, a))
    
    # A copy of the original image
    img_with_stickers = img_bgr.copy()

    for face in faces:
        landmarks = face_landmarking(img_bgr, face)

        # the landmarks 36 to 41 are for the left eye, and 42 to 47 are for the right eye
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        # Calculate the center points of the eyes
        left_eye_center = (sum([p.x for p in left_eye]) // len(left_eye), sum([p.y for p in left_eye]) // len(left_eye))
        right_eye_center = (sum([p.x for p in right_eye]) // len(right_eye), sum([p.y for p in right_eye]) // len(right_eye))

        # Calculate the angle of tilt
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = math.degrees(math.atan2(dy, dx))  # Angle in degrees

        # Calculate the bounding box for the glasses based on the eye landmarks
        glasses_width = int(abs(left_eye_center[0] - right_eye_center[0]) * 2)
        glasses_height = int(glasses_width * glasses_bgra.height / glasses_bgra.width)

        # Resize and rotate the glasses image
        resized_glasses = glasses_bgra.resize((glasses_width, glasses_height))
        rotated_glasses = resized_glasses.rotate(-0.8*angle, expand=True, resample=Image.BICUBIC)  # Negative angle to correct orientation

        # Calculate the position for the glasses, adjusting for the rotation
        x1 = left_eye_center[0] - int(0.25 * glasses_width)
        y1 = min(left_eye_center[1], right_eye_center[1]) - int(0.45 * glasses_height)

        # Convert PIL image to NumPy array
        glasses_np = np.array(rotated_glasses)
        
        # Overlay the glasses on the image
        img_with_stickers = overlay_transparent(img_with_stickers, glasses_np, x1, y1)
        
    return img_with_stickers



def add_noses_sticker(img_bgr, sticker_path, faces):
    nose_pil = Image.open(sticker_path)

    # Check the color mode and convert to RGBA    
    nose_rgba = nose_pil.convert('RGBA')
    
    # Convert the nose_rgba to BGRA
    r, g, b, a = nose_rgba.split()
    nose_bgra = Image.merge("RGBA", (b, g, r, a))

    # A copy of the original image
    img_with_stickers = img_bgr.copy()

    for face in faces:
        landmarks = face_landmarking(img_bgr, face)

        # Assuming that the landmarks 27 to 35 are for the nose area
        nose_area = [landmarks.part(i) for i in range(27, 36)]

        # Calculate the bounding box for the nose based on the nose landmarks
        nose_width = int(abs(nose_area[0].x - nose_area[-1].x) * 2.1)
        nose_height = int(nose_width * nose_bgra.height / nose_bgra.width)

        # the landmarks 31 and 35 are the leftmost and rightmost points of the nose area
        nose_left = landmarks.part(31)
        nose_right = landmarks.part(35)

        # Calculate the center point of the nose
        nose_center_x = (nose_left.x + nose_right.x) // 2

        nose_top = landmarks.part(27)  # Use 28 if it's more accurate 
        nose_bottom = landmarks.part(33)

        # Calculate the midpoint of the vertical length of the nose
        nose_center_y = (nose_top.y + nose_bottom.y) // 2

        # Calculate the angle of tilt using the eyes as reference
        left_eye_indices = range(36, 42)
        right_eye_indices = range(42, 48)
        angle = calculate_eye_angle(landmarks, left_eye_indices, right_eye_indices)

        # Resize the nose image
        resized_nose_pil = nose_bgra.resize((nose_width, nose_height))

        rotated_nose = resized_nose_pil.rotate(-angle, expand=True, resample=Image.BICUBIC)


        # the position for the nose
        x1 = nose_center_x - (nose_width // 2)
        y1 = nose_center_y - (nose_height // 2)+ int(0.1 * nose_height)  # Adding a slight downward offset
        # Convert PIL image to NumPy array
        nose_np = np.array(rotated_nose)
        
        # Overlay the nose on the image
        img_with_stickers = overlay_transparent(img_with_stickers, nose_np, x1, y1)
        
    return img_with_stickers

def add_animal_faces_sticker(img_bgr, sticker_path, faces):
    animal_face_pil = Image.open(sticker_path)

    # Check the color mode and convert to RGBA
    animal_face_rgba = animal_face_pil.convert('RGBA')
    
    # Convert the animal_face_rgba to BGRA
    r, g, b, a = animal_face_rgba.split()
    animal_face_bgra = Image.merge("RGBA", (b, g, r, a))
    
    # A copy of the original image
    img_with_stickers = img_bgr.copy()

    for face in faces:
        landmarks = face_landmarking(img_bgr, face)

        # Find the top of the forehead using landmarks above the eyes
        # Assuming landmarks 19 to 24 represent the eyebrows
        forehead_top = min(landmarks.part(i).y for i in range(68, 81)) 

        # Calculate the center point between the eyes as an anchor
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        eye_center_x = (left_eye[0].x + right_eye[3].x) // 2
        eye_center_y = (left_eye[3].y + right_eye[0].y) // 2

        # Calculate the size of the animal face sticker based on the width between the temples
        head_width = int(abs(landmarks.part(0).x - landmarks.part(16).x)*1.1) #1.5
        head_height = int(head_width * animal_face_bgra.height / animal_face_bgra.width)

        # Resize the animal face sticker
        resized_animal_face = animal_face_bgra.resize((head_width, head_height))

        # Calculate the position for the animal face sticker
        x1 = eye_center_x - (head_width // 2)
        y1 = forehead_top - int(0.18 * head_height)

        # Convert PIL image to NumPy array
        animal_face_np = np.array(resized_animal_face)

        # Overlay the animal face on the image
        img_with_stickers = overlay_transparent(img_with_stickers, animal_face_np, x1, y1)

    return img_with_stickers


# Function to process the image
def process_image(image, sticker_choice):
    if sticker_choice:
        # add .png to the sticker_choice
        sticker_name = sticker_choice + '.png'
        # find sticker's category
        sticker_category = STICKER_TO_CATEGORY[sticker_name]
        # Path to the single sticker
        sticker_path = os.path.join('stickers',sticker_category, sticker_name)

        # Convert PIL image to OpenCV format BGR
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = face_detecting(image_bgr)
        # print(sticker_category)
        if sticker_category == 'ears':
            img_with_stickers = add_ears_sticker(image_bgr, sticker_path, faces)
        elif sticker_category == 'glasses':
            img_with_stickers = add_glasses_sticker(image_bgr, sticker_path, faces)
        elif sticker_category == 'noses':
            img_with_stickers = add_noses_sticker(image_bgr, sticker_path, faces)
        elif sticker_category == 'headbands':
            img_with_stickers = add_ears_sticker(image_bgr, sticker_path, faces)
        else:
            img_with_stickers = add_glasses_sticker(image_bgr, sticker_path, faces)
        # Convert back to PIL image
        img_with_stickers_pil = Image.fromarray(cv2.cvtColor(img_with_stickers, cv2.COLOR_BGR2RGB))
        return img_with_stickers_pil
    else:
        return image

def process_image_with_selections(image_input):
    # Convert PIL image to OpenCV format BGR
    image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    # Detect faces
    faces = face_detecting(image_bgr)
    
    # A copy of the original image to apply stickers on
    img_with_stickers = image_bgr.copy()

    for category, sticker_name in sticker_selections.items():
        if sticker_name:  # Check if a sticker was selected in this category
            # the sticker file path
            sticker_path = os.path.join('stickers', category, sticker_name + '.png')

            # Apply the selected sticker based on its category
            if category == 'ears':
                img_with_stickers = add_ears_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'glasses':
                img_with_stickers = add_glasses_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'noses':
                img_with_stickers = add_noses_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'headbands':
                img_with_stickers = add_hats_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'hats':
                img_with_stickers = add_hats_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'animal face':
                img_with_stickers = add_animal_faces_sticker(img_with_stickers, sticker_path, faces)
            else:
                img_with_stickers = img_with_stickers
    # Convert back to PIL image
    img_with_stickers_pil = Image.fromarray(cv2.cvtColor(img_with_stickers, cv2.COLOR_BGR2RGB))
    
    print("Selected stickers:")
    for category, selection in sticker_selections.items():
        print(f"{category}: {selection}")

    return img_with_stickers_pil

# This dictionary will hold the user's sticker selections
sticker_selections = {}

# Function to update sticker selections
def update_selections(category, selection):
    # sticker_selections[category] = selection
    sticker_selections[category] = None if selection == "None" else selection
    return ""

# Function to load an example image
def load_example_image(image_path):
    return gr.Image.from_file(image_path)

from PIL import Image

def resize_image(image, target_width, target_height):
    # Maintain aspect ratio
    original_width, original_height = image.size
    ratio = min(target_width/original_width, target_height/original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    # Use Image.LANCZOS for high-quality downsampling
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def get_face_crops(image_bgr, faces, target_width=500, target_height=130):
    face_crops = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_crop = image_bgr[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        # Resize image to fit the display while maintaining aspect ratio
        resized_face = resize_image(face_pil, target_width, target_height)
        face_crops.append(resized_face)
    return face_crops


def get_face_crops2(image_bgr, faces):
    # Convert color space from BGR to RGB since OpenCV uses BGR by default
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_crops = []
    for face in faces:
        # Extract the region of interest (the face) from the original image
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_crop = image_rgb[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_crop)
        face_crops.append(face_pil)
    return face_crops

# Function to process uploaded images and display face crops
def process_and_show_faces(image_input):
    # Convert PIL image to OpenCV format BGR
    image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    # Detect faces
    faces = face_detecting(image_bgr)
    # Get individual face crops
    face_crops = get_face_crops(image_bgr, faces)
    # Return face crops to display them in the interface
    return face_crops


face_outputs = []
for i in range(MAX_EXPECTED_FACES):
    face_output = gr.Image(label=f"Face {i+1}")
    face_outputs.append(face_output)

# This list will hold the Checkbox components for each face
checkboxes = []

def process_selected_faces(image_input, selected_face_indices):
    # Convert PIL image to OpenCV format BGR
    image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    # Detect all faces
    all_faces = face_detecting(image_bgr)

    # Filter faces to get only those selected
    faces = [all_faces[i] for i in selected_face_indices]

    img_with_stickers = image_bgr.copy()

    for category, sticker_name in sticker_selections.items():
        if sticker_name:  # Check if a sticker was selected in this category
            # the sticker file path
            sticker_path = os.path.join('stickers', category, sticker_name + '.png')

            # Apply the selected sticker based on its category
            if category == 'ears':
                img_with_stickers = add_ears_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'glasses':
                img_with_stickers = add_glasses_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'noses':
                img_with_stickers = add_noses_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'headbands':
                img_with_stickers = add_hats_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'hats':
                img_with_stickers = add_hats_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'animal face':
                img_with_stickers = add_animal_faces_sticker(img_with_stickers, sticker_path, faces)
            else:
                img_with_stickers = img_with_stickers
    # Convert back to PIL image
    img_with_stickers_pil = Image.fromarray(cv2.cvtColor(img_with_stickers, cv2.COLOR_BGR2RGB))
    
    print("Selected stickers:")
    for category, selection in sticker_selections.items():
        print(f"{category}: {selection}")

    return img_with_stickers_pil

def handle_face_selection(image_input, *checkbox_states):
    selected_face_indices = [i for i, checked in enumerate(checkbox_states) if checked]
    print("selected_face_indices:",selected_face_indices)
    return process_selected_faces(image_input, selected_face_indices)

def update_interface_with_faces(image_input):
    image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    faces = face_detecting(image_bgr)
    face_crops = get_face_crops(image_bgr, faces)
    return [(face, f"Face {i+1}") for i, face in enumerate(face_crops)]

def detect_and_display_faces(image_input):
    image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    faces = face_detecting(image_bgr)  
    face_crops = get_face_crops(image_bgr, faces)  
    if not face_crops:
        # Return empty images and unchecked boxes if no faces are detected
        return [None] * MAX_EXPECTED_FACES + [False] * MAX_EXPECTED_FACES
    # Return face crops and True for each checkbox to indicate they should be checked
    # Pad the list with None and False if fewer faces than MAX_EXPECTED_FACES are detected
    output = face_crops + [None] * (MAX_EXPECTED_FACES - len(face_crops))
    output += [True] * len(face_crops) + [False] * (MAX_EXPECTED_FACES - len(face_crops))
    return output


# Create the Gradio interface
with gr.Blocks() as demo:
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Original Image")
        with gr.Column():
            output_image = gr.Image(label="Image with Stickers")
        # Prepare the checkboxes and image placeholders
    detect_faces_btn = gr.Button("Detect Faces")

    with gr.Row():
        face_checkboxes = [gr.Checkbox(label=f"Face {i+1}") for i in range(7)]

    with gr.Row():
        face_images = [gr.Image(height=150, width=100, min_width=30) for i in range(7)]
    
    detect_faces_btn.click(
        detect_and_display_faces,
        inputs=[image_input],
        outputs=face_images + face_checkboxes  
    )

    process_button = gr.Button("Apply Stickers To Selected Faces")

    process_button.click(
        handle_face_selection, 
        inputs=[image_input] + face_checkboxes, 
        outputs=output_image
    )

    # Iterate over each category to create a row for the category
    for category, stickers in STICKER_PATHS.items():
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"## {category}")  # Category label
            with gr.Column(scale=10):      
                # Iterate over stickers in sets of 10
                for i in range(0, len(stickers), 10):
                    with gr.Row():
                        for sticker_path in stickers[i:i+10]:
                            gr.Image(value=sticker_path, height=130, width=500, min_width=30, interactive=False, show_download_button=False, container=False)  # Sticker image
                with gr.Row():
                    # radio = gr.Radio(label=' ', choices=[stickers[i].split('/')[-1].replace('.png', '') for i in range(len(stickers))], container=False, min_width=50)
                    choices = ["None"] + [sticker.split('/')[-1].replace('.png', '') for sticker in stickers]
                    radio = gr.Radio(label=' ', choices=choices, value="None", container=False, min_width=50)
                    radio.change(lambda selection, cat=category: update_selections(cat, selection), inputs=[radio], outputs=[])
    # Button to apply all selected stickers
    apply_btn = gr.Button("Apply Stickers To All")
    apply_btn.click(process_image_with_selections, inputs=[image_input], outputs=output_image)

    # # List of example images
    # examples = [
    #     ["./example_images/Herminone.jpg"],
    #     ["./example_images/3-people.jpg"],
    #     ["./example_images/multi.png"]
    # ]

    # # Adding an Examples component to allow users to load example images
    # demo.examples(examples, inputs=[image_input])
demo.launch()


