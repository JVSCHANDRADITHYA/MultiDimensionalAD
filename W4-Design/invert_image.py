# invert an transperant background image
from PIL import Image
def invert_image(input_path, output_path):
    # Open the image file
    img = Image.open(input_path).convert("RGBA")
    
    # Invert the colors
    r, g, b, a = img.split()
    r = r.point(lambda i: 255 - i)
    g = g.point(lambda i: 255 - i)
    b = b.point(lambda i: 255 - i)
    
    # Merge back the channels
    inverted_img = Image.merge('RGBA', (r, g, b, a))
    
    # Save the inverted image
    inverted_img.save(output_path)

# Example usage
invert_image(r"C:\Users\adith\Downloads\5576886.png", 'inverted_image.png')
