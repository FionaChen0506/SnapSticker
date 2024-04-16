# from PIL import Image
# # Load the image of the cat headband
# img_path_cat_headband = './stickers/ears/white-bunny-ears.png'
# img_cat_headband = Image.open(img_path_cat_headband)

# # Crop the image to remove transparent spaces
# bbox_cat_headband = img_cat_headband.getbbox()

# # Cropping the image to the bounding box
# cropped_img_cat_headband = img_cat_headband.crop(bbox_cat_headband)

# cropped_img_cat_headband_path = './cropped_cat-headband.png'
# cropped_img_cat_headband.save(cropped_img_cat_headband_path)
# cropped_img_cat_headband_path

from PIL import Image

img_path_cat_headband = './stickers/ears/white-bunny-ears.png'
img_cat_headband = Image.open(img_path_cat_headband)

# Get the bounding box to remove transparent spaces
bbox_cat_headband = img_cat_headband.getbbox()

if bbox_cat_headband:  # Checking if bounding box is not None
    left, upper, right, lower = bbox_cat_headband
    new_lower = lower - 30  # Reducing the lower boundary by 10 pixels
    new_bbox_cat_headband = (left, upper, right, new_lower)

    # Cropping the image to the new bounding box
    cropped_img_cat_headband = img_cat_headband.crop(new_bbox_cat_headband)

    # Save the cropped image
    cropped_img_cat_headband_path = './cropped_cat-headband.png'
    cropped_img_cat_headband.save(cropped_img_cat_headband_path)


