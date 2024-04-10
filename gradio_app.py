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

        # Assuming that the landmarks 68 to 80 are for the forehead
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
        
        # Add ear stickers
        img_with_stickers = add_ears_sticker(image_bgr, sticker_path, faces)

        # Convert back to PIL image
        img_with_stickers_pil = Image.fromarray(cv2.cvtColor(img_with_stickers, cv2.COLOR_BGR2RGB))
        return img_with_stickers_pil
    else:
        return image





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
                    radio = gr.Radio(label=' ', choices=[stickers[i].split('/')[-1].replace('.png', '') for i in range(len(stickers))], container=False, min_width=50)
                    radio.change(process_image, inputs=[image_input, radio], outputs=output_image)

demo.launch()


