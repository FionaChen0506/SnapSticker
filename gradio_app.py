import gradio as gr
import cv2
import numpy as np
from PIL import Image
import dlib
import os

from constants import *

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

        # Calculate the bounding box for the ears based on the eye landmarks
        ears_width = int(abs(forehead[0].x - forehead[-1].x) * 2) 
        ears_height = int(ears_width * ears_bgra.height / ears_bgra.width)

        # Resize the ears image
        resized_ears_pil = ears_bgra.resize((ears_width, ears_height))
        
        # Calculate the position for the ears
        y1 = min([point.y for point in forehead]) - int(0.6 * ears_width)
        y2 = y1 + ears_height
        x1 = forehead[0].x - int(0.2 * ears_width) 
        x2 = x1 + ears_width

        # Convert PIL image to NumPy array
        ears_np = np.array(resized_ears_pil)
        
        # Overlay the ears on the image
        img_with_stickers = overlay_transparent(img_with_stickers, ears_np, x1, y1)
        
    return img_with_stickers

# # Function to add hats stickers
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

        #the landmarks 36 to 41 are for the left eye, and 42 to 47 are for the right eye
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        # Calculate the center point between the eyes
        left_eye_center = ((left_eye[0].x + left_eye[3].x) // 2, (left_eye[0].y + left_eye[3].y) // 2)
        right_eye_center = ((right_eye[0].x + right_eye[3].x) // 2, (right_eye[0].y + right_eye[3].y) // 2)
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2

        # Calculate the size of the hat based on the width between the eyes
        hat_width = int(abs(left_eye[0].x - right_eye[3].x) * 1.5)
        hat_height = int(hat_width * hat_bgra.height / hat_bgra.width)

        # Resize the hat image
        resized_hat = hat_bgra.resize((hat_width, hat_height))

        # Calculate the position for the hat
        # y1 = eye_center_y - hat_height  # Placing the bottom of the hat at eye level
        y1 = eye_center_y - hat_height - int(0.3 * hat_height)
        x1 = eye_center_x - (hat_width // 2)  # Centering the hat on the midpoint between the eyes

        # Convert PIL image to NumPy array
        hat_np = np.array(resized_hat)

        # Overlay the hat on the image
        img_with_stickers = overlay_transparent(img_with_stickers, hat_np, x1, y1)

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
        left_eye = [landmarks.part(i) for i in range(37, 43)]
        right_eye = [landmarks.part(i) for i in range(43, 49)]

        # Calculate the bounding box for the glasses based on the eye landmarks
        glasses_width = int(abs(left_eye[0].x - right_eye[3].x)*1.9)
        glasses_height = int(glasses_width * glasses_bgra.height / glasses_bgra.width)


        # Resize the glasses image
        resized_glasses = glasses_bgra.resize((glasses_width, glasses_height))
        
        # Calculate the position for the glasses
        y1 = min([point.y for point in left_eye + right_eye]) - int(0.4 * glasses_height)
        y2 = y1 + glasses_height
        x1 = (left_eye[0].x) - int(0.2 * glasses_width)
        x2 = x1 + glasses_width

        # Convert PIL image to NumPy array
        glasses_np = np.array(resized_glasses)
        
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
        nose_width = int(abs(nose_area[0].x - nose_area[-1].x) * 1.8)
        nose_height = int(nose_width * nose_bgra.height / nose_bgra.width)

        # Resize the nose image
        resized_nose_pil = nose_bgra.resize((nose_width, nose_height))
        
        # Calculate the position for the nose
        y1 = min([point.y for point in nose_area]) + int(0.3 * nose_height)
        y2 = y1 + nose_height
        x1 = min([point.x for point in nose_area]) - int(0.1 * nose_width)
        x2 = x1 + nose_width

        # Convert PIL image to NumPy array
        nose_np = np.array(resized_nose_pil)
        
        # Overlay the nose on the image
        img_with_stickers = overlay_transparent(img_with_stickers, nose_np, x1, y1)
        
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
    # print("Selected stickers:")
    # for category, selection in sticker_selections.items():
    #     print(f"{category}: {selection}")

    # return image
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
                img_with_stickers = add_ears_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'hats':
                img_with_stickers = add_hats_sticker(img_with_stickers, sticker_path, faces)
            elif category == 'animal face':
                img_with_stickers = add_glasses_sticker(img_with_stickers, sticker_path, faces)
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

# Create the Gradio interface
with gr.Blocks() as demo:
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Original Image")
        with gr.Column():
            output_image = gr.Image(label="Image with Stickers")
    

        
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
    apply_btn = gr.Button("Apply Stickers")
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


