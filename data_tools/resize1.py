import os
from PIL import Image

def convert_images(input_dir='archive/Dataset', output_dir='archive/Dataset_48x48_bw'):
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each emotion label directory
    for label in os.listdir(input_dir):
        label_input_path = os.path.join(input_dir, label)
        if os.path.isdir(label_input_path):
            # Mirror the folder structure in the output directory
            label_output_path = os.path.join(output_dir, label)
            os.makedirs(label_output_path, exist_ok=True)

            # Process each image file in the label directory
            for filename in os.listdir(label_input_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    input_path = os.path.join(label_input_path, filename)
                    output_path = os.path.join(label_output_path, filename)
                    try:
                        with Image.open(input_path) as img:
                            # Convert to grayscale (black and white)
                            bw = img.convert('L')
                            # Resize to 48x48 pixels using high-quality resampling
                            resized = bw.resize((48, 48), resample=Image.LANCZOS)
                            # Save the processed image
                            resized.save(output_path)
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")

if __name__ == '__main__':
    convert_images()
