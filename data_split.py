import os
import shutil

# Define the source folder containing PNG and XML files
source_folder = r'place your folder name'
# Define the destination folders
images_folder = 'images'
annotations_folder = 'annotations'

# Define the image file extensions you want to handle
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# Loop through the files in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(image_extensions):
        # Move PNG image to images folder
        shutil.move(os.path.join(source_folder, filename), os.path.join(images_folder, filename))

    if filename.endswith('.xml'):
        shutil.move(os.path.join(source_folder, filename), os.path.join(annotations_folder, filename))
            

print("Files have been moved successfully.")
