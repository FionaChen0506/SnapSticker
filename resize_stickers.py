from PIL import Image

def resize_image(input_path, output_path, scale_factor):
    # Open an image file
    with Image.open(input_path) as img:
        # Calculate the new size
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * 1.0)
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save the resized image
        resized_img.save(output_path, format='PNG')

# Example usage
input_path = './stickers/hats/graduation-hat.png'
output_path = './new-graduation-hat.png'
scale_factor = 1.5

resize_image(input_path, output_path, scale_factor)
