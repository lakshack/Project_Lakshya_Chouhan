Project Proposal: Face Mask Detection Using Deep Learning

Problem Description
With the widespread outbreak of COVID-19, wearing face masks in public became a crucial
safety measure to prevent the spread of the virus. However, enforcing mask compliance in
crowded areas such as hospitals, shopping malls, and transportation hubs poses a significant
challenge. Automated face mask detection using deep learning provides an efficient solution for
monitoring and ensuring compliance with mask mandates.

This project aims to develop a deep learning model that can classify images of human faces into
two categories: With Mask and Without Mask. The model will be trained on a dataset of facial
images featuring people wearing masks correctly and those without masks. By leveraging deep
learning techniques, this system can be integrated into surveillance cameras or mobile
applications to facilitate automated monitoring in real-time environments.

Problem Formulation
Input: An RGB image of a human face.
Output: A binary classification output indicating whether the person is wearing a mask (With
Mask) or not (Without Mask).

Data Source
The dataset used for training is the “Face Mask Dataset” on kaggle by Omkar gurav consists of
more than 7000 labeled images with three color channels (RGB). It includes a diverse range of
images featuring different lighting conditions, facial orientations, occlusions, and mask types.
This dataset ensures robustness and generalization to real-world scenarios. The data will be
split into training, validation, and testing sets for effective model evaluation.

Model Architecture Choice
A custom Convolutional Neural Network (CNN) architecture is chosen for this task due to its
effectiveness in image classification. CNNs are well-suited for processing visual data, capturing
spatial features, and recognizing patterns in images. The architecture consists of multiple
convolutional layers with ReLU activation functions, followed by max-pooling layers to reduce
dimensionality while preserving essential features. Fully connected dense layers are used for
final classification, with a softmax activation function in the output layer to differentiate between
the two classes.

To improve generalization and prevent overfitting, dropout layers are incorporated. The model is
optimized using the Adam optimizer, and the loss function used is categorical cross-entropy.
Data augmentation techniques such as random rotation, horizontal flipping, and brightness
adjustments are applied to improve robustness against variations in real-world images.

Accuracy metrics
The model will be evaluated using Accuracy, Precision, Recall, and F1-Score. Accuracy measures the overall correctness of predictions. Precision and Recall assess the model's ability to identify faces with and without masks, while F1-Score balances both. Additionally, a Confusion Matrix will provide detailed insights into prediction performance, and the AUC-ROC curve will measure the model's ability to distinguish between the two classes.
