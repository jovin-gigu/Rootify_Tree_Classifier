import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("place_detection_model.h5")

# Define class labels for 10 trees
class_labels = [
    "(Norfolk Island Pine)", "(Mango tree)", "(Neem Tree)", "(Silver Date Palm)", "(Coconut Palm)", 
    "(Tabebuia aurea)", "(Mango Tree)", "(Araucaria)", "(Tamarind Tree)", "(Bamboo)"
]


# Load and preprocess the test image
img_path = "test1.jpg"  # Change to your test image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(img_array)

# Get the predicted class
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_label = class_labels[predicted_class_index]

# Display the result
plt.imshow(img)
plt.title(f"Predicted Class: {predicted_label}")
plt.axis("off")
plt.show()

# Print confidence scores
print("\nPrediction Probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"{class_labels[i]}: {prob:.4f}")
