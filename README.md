# Classification-d-images-d-articles-de-mode
# Image Classification Project with Fashion MNIST Dataset

This project demonstrates image classification using the Fashion MNIST dataset with TensorFlow/Keras. The code includes the following key components:

## 1. Data Loading and Preprocessing
- Load the Fashion MNIST dataset using TensorFlow/Keras.
- Convert image format to floating-point and normalize pixel values to the range [0, 1].

## 2. Data Exploration
- Display the shape of the dataset.
- Visualize the 500th image in grayscale.

## 3. Neural Network Model
- Build a simple neural network using the Sequential API.
- Flatten the data and add dense layers with ReLU activation.
- Compile the model using sparse categorical crossentropy as the loss function, Adam optimizer, and accuracy as the metric.

## 4. Data Augmentation
- Utilize `ImageDataGenerator` for data augmentation, including rotation, width and height shifts, shear transformations, zooming, and horizontal flipping.

## 5. Training and Evaluation
- Train the neural network with augmented data for 10 epochs, monitoring validation performance.
- Plot training and validation loss and accuracy curves.
- Evaluate the model on the test set and display test loss and accuracy.

## 6. Convolutional Neural Network (CNN)
- Build a CNN model with convolutional and pooling layers for more complex feature extraction.
- Compile and train the CNN model with the same data augmentation strategy for 5 epochs.
- Visualize the training and validation performance of the CNN model.

## 7. Confusion Matrix
- Generate predictions using the CNN model on the test set.
- Display a confusion matrix to visualize the model's performance across different classes.

## Dependencies
- matplotlib
- tensorflow
- numpy
- pandas
- scikit-learn
- seaborn

## How to Run
1. Ensure you have the required dependencies installed (`pip install matplotlib tensorflow numpy pandas scikit-learn seaborn`).
2. Run the code in a Python environment.

Feel free to customize and extend the code based on your specific requirements. If you have any questions or need further assistance, please reach out.

Happy coding!
