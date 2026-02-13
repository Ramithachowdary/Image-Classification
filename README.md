# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion MNIST dataset. The Fashion MNIST dataset contains images of clothing items such as T-shirts, trousers, dresses, and shoes, and the model aims to classify them into their respective categories correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model
<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/3dd19a0a-67f9-4971-8d9f-57e905dbc5f2" />

## DESIGN STEPS
### STEP 1: Problem Statement
Define the objective of classifying fashion items (such as shirts, shoes, bags, etc.) using a Convolutional Neural Network (CNN).

### STEP 2: Dataset Collection
Use the Fashion MNIST dataset, which contains 60,000 training images and 10,000 test images of labeled clothing and accessory items.

### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 4: Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers tailored for 10 fashion categories.

### STEP 5: Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 6: Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: Ramitha chowdary S
### Register Number: 212224240130 
```python
# Import Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Print Name and Register Number
print("=" * 50)
print("Name: Ramitha chowdary S")
print("Register Number: 212224240130")
print("=" * 50)
print()

# Load Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Preprocess Dataset
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display Model Summary
print("\nModel Architecture:")
model.summary()
print()

# Train Model
print("Training the model...")
history = model.fit(train_images, train_labels,
                    epochs=5,
                    batch_size=32,
                    validation_data=(test_images, test_labels),
                    verbose=1)
print()

# Plot Training vs Validation Loss
print("Name: Ramitha chowdary S")
print("Register Number: 212224240130")
print()

plt.figure(figsize=(10, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
print()

# Evaluate Model
print("Name: Ramitha chowdary S")
print("Register Number: 212224240130")
print()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print()

# Make Predictions
predictions = model.predict(test_images, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# Define Class Names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot Confusion Matrix
print("Name: Ramitha chowdary S")
print("Register Number: 212224240130")
print()

cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
print()

# Print Classification Report
print("Name: Ramitha chowdary S")
print("Register Number: 212224240130")
print()
print("Classification Report:")
print("=" * 70)
print(classification_report(test_labels, y_pred,
                            target_names=class_names))
print()

# Display Sample Prediction
print("Name: Ramitha chowdary S")
print("Register Number: 212224240130")
print()

index = 50
plt.figure(figsize=(5, 5))
plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
plt.title(f"Actual: {class_names[test_labels[index]]}\nPredicted: {class_names[y_pred[index]]}")
plt.axis("off")
plt.tight_layout()
plt.show()

print(f"Sample Image #{index}")
print(f"Actual: {class_names[test_labels[index]]}")
print(f"Predicted: {class_names[y_pred[index]]}")
print()
```

## OUTPUT
### Model Architecture
<img width="572" height="445" alt="image" src="https://github.com/user-attachments/assets/4ddc722c-a567-40f2-afdf-ed1fa5a754d8" />

### Training Loss per Epoch
<img width="840" height="553" alt="image" src="https://github.com/user-attachments/assets/871b2d3a-4ab3-4b8b-8cf0-1cd49e6f7440" />
<img width="226" height="90" alt="image" src="https://github.com/user-attachments/assets/c1aa896e-bd60-4529-9207-73e1c5ee20c9" />

### Confusion Matrix

<img width="805" height="711" alt="image" src="https://github.com/user-attachments/assets/1de18788-867a-4ed4-844b-5aadc906f229" />


### Classification Report

<img width="485" height="332" alt="image" src="https://github.com/user-attachments/assets/be8a01e4-bc66-4521-984a-6939fab9daa0" />

### New Sample Data Prediction

<img width="407" height="507" alt="image" src="https://github.com/user-attachments/assets/9f811f72-7b1c-489c-84a5-440f57fca5e4" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
