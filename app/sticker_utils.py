import cv2
import dlib
from PIL import Image
import numpy as np
import os

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

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

def add_ear_stickers(image_path):
    # # Load ears image with alpha channel using PIL
    # ears_pil = Image.open('/Users/chenshaoyujie/Desktop/CS5330/SnapSticker/app/static/stickers/ears/fluffy-bunny-ear.png')
    # ears_rgba = ears_pil.convert('RGBA')
    
    # # Load the image
    # image = cv2.imread(image_path)
    
    # # Copy of the original image
    # img_with_stickers = image.copy()

    # # Detect faces in the image
    # faces = detector(image)

    # for face in faces:
    #     landmarks = predictor(image, face)

    #     # Assuming that the landmarks 68 to 80 are for the forehead
    #     forehead = [landmarks.part(i) for i in range(68, 81)]

    #     # Calculate the bounding box for the ears based on the eye landmarks
    #     ears_width = int(abs(forehead[0].x - forehead[-1].x) * 2) 
    #     ears_height = int(ears_width * ears_rgba.height / ears_rgba.width)

    #     # Resize the ears image
    #     resized_ears_pil = ears_rgba.resize((ears_width, ears_height))

    #     # Calculate the position for the ears
    #     y1 = min([point.y for point in forehead]) - int(0.6 * ears_width)
    #     y2 = y1 + ears_height
    #     x1 = forehead[0].x - int(0.2 * ears_width) 
    #     x2 = x1 + ears_width

    #     # Convert PIL image to NumPy array
    #     ears_np = np.array(resized_ears_pil)

    #     # Overlay the ears on the image
    #     img_with_stickers = overlay_transparent(img_with_stickers, ears_np, x1, y1)
    
    # # Save the modified image
    # modified_image_path = os.path.join('static', 'uploads', 'modified_' + image_path.split('/')[-1])
    # # cv2.imwrite(modified_image_path, img_with_stickers)
    # # img = Image.open(modified_image_path)
    # # img.show()
    return image_path
