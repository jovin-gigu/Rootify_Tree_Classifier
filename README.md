# 🌳 Rootify: Tree Classifier

**Rootify** is a primitive yet effective tree species classifier developed using images of trees from our college campus. It uses image processing and deep learning techniques—specifically transfer learning with a fine-tuned VGG16 model—to detect and classify different types of trees from input images.

---

## 🚀 Features

- 🖼️ **Image Preprocessing**: Resizes, normalizes, and augments images to enrich the training dataset.
- 🤖 **Transfer Learning**: Uses a VGG16-based model fine-tuned on our custom tree dataset for better accuracy.
- 🌲 **Species Prediction**: Classifies new tree images and visualizes prediction results.
- 📈 **Multi-Class Support**: Recognizes multiple tree species found on campus.

---

## 📁 Repository Structure
```bash
      Rootify_Tree_Classifier/
      │
      ├── images/                    # Raw images (organized by class, e.g., tree1/, tree2/, ...)
      ├── NN_images_processed/       # Preprocessed & augmented images
      ├── preprocess.py              # Script to preprocess and augment images
      ├── train.py                   # Script to train the classifier (VGG16-based)
      ├── test.py                    # Script to test the trained model on new images
      ├── tree_classification_model.h5  # Trained model file
      ├── requirements.txt           # Python dependencies
      └── README.md                  # This file!
```
---

## 📦 Installation

1. Clone this repository:
```bash
   git clone https://github.com/yourusername/Rootify_Tree_Classifier.git
   cd Rootify_Tree_Classifier
```
Install dependencies:
```bash
   pip install -r requirements.txt


