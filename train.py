import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Define paths
dataset_path = "images_processed"  # Your dataset folder

# Image parameters
img_size = (224, 224)
batch_size = 8  # Smaller batch size since dataset is small
epochs = 20  # Start with 20, fine-tuning will continue later

# Check number of classes dynamically
num_classes = len(os.listdir(dataset_path))  # Count the number of class folders

# 🔹 **Enhanced Data Augmentation**
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    fill_mode='nearest',
    validation_split=0.2  # 80% training, 20% validation
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 🔹 **Load Pre-trained VGG16 model**
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze convolutional layers for now

# 🔹 **Add Custom Layers**
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)  # Matches the number of classes

# Define Model
model = Model(inputs=base_model.input, outputs=output)

# 🔹 **Compile Model**
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🔹 **Callbacks to prevent overfitting**
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# 🔹 **Train Model (Phase 1)**
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# 🔹 **Fine-tuning: Unfreeze last layers for extra learning**
base_model.trainable = True
for layer in base_model.layers[:10]:  # Keep first 10 layers frozen
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🔹 **Train again with fine-tuning (Phase 2)**
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Fine-tune for 10 more epochs
    callbacks=[early_stopping, reduce_lr],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# 🔹 **Save the final trained model**
model.save("tree_classification_model.h5")

print("✅ Training complete! Model saved as 'tree_classification_model.h5'.")
