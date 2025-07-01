import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# Define paths
DATASET_PATH = 'images'  # Your dataset folder
PROCESSED_PATH = 'NN_images_processed'  # Where processed images will be stored
IMG_SIZE = (224, 224)  # Resize all images to 224x224

# Create processed dataset directory if it doesn't exist
if not os.path.exists(PROCESSED_PATH):
    os.makedirs(PROCESSED_PATH)

# Data Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Process each tree type (class)
for class_folder in sorted(os.listdir(DATASET_PATH), key=lambda x: int(x.replace("tree", ""))):
    class_path = os.path.join(DATASET_PATH, class_folder)

    # ✅ **Fix: Skip files and ensure only directories are processed**
    if not os.path.isdir(class_path):
        print(f"⚠️ Skipping '{class_path}', not a directory.")
        continue  # Skip files like 'place_detection_model.h5'

    processed_class_path = os.path.join(PROCESSED_PATH, class_folder)
    
    if not os.path.exists(processed_class_path):
        os.makedirs(processed_class_path)
    
    # Process each image in the class folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        processed_img_path = os.path.join(processed_class_path, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping '{img_path}', couldn't read.")
            continue
        
        # Resize and normalize
        img = cv2.resize(img, IMG_SIZE)
        img = img_to_array(img) / 255.0  # Normalize to [0,1]
        
        # Save original preprocessed image
        cv2.imwrite(processed_img_path, (img * 255).astype(np.uint8))
        
        # Expand dimensions for augmentation
        img = np.expand_dims(img, axis=0)
        
        # Apply augmentation & save 5 variations
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=processed_class_path, 
                                  save_prefix=f'aug_{i}', save_format='jpg'):
            i += 1
            if i >= 5:  # Generate 5 augmented images per original
                break
        
print("✅ Preprocessing complete! Augmented images saved in 'images_processed'.")
