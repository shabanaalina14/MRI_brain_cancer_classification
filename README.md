Brain_cancer_classification Image Classification for Multi-Class Dataset

Project Overview

In this project, I developed an image classification model to categorize images into one of four classes using Convolutional Neural Networks (CNNs). The dataset comprises images from four distinct categories.

Data Collection & Preprocessing

I sourced the training and testing images from respective folders. Each image was resized to a consistent size of 160x160 pixels to ensure uniformity. Data augmentation techniques, such as rotation, shifting, zooming, and horizontal flipping, were applied to increase the dataset's variability and improve model generalization.

Model Architecture

Constructed a CNN model consisting of multiple convolutional layers followed by max-pooling layers. Batch normalization was used to standardize the inputs of each layer, enhancing the model's stability and speeding up training. Dropout layers were added to prevent overfitting by randomly setting a fraction of input units to zero during training. The model concludes with a dense layer with a softmax activation function to output probabilities for each class.

Training & Optimization

The model was trained using the Adam optimizer with a learning rate of 0.00007. Early stopping was implemented to halt training if the validation loss did not improve after a set number of epochs, preventing overfitting and saving computational resources. Sparse categorical crossentropy was used as the loss function since our labels are integers.

Evaluation

The model's performance was evaluated using both training and validation datasets. Training and validation accuracy over epochs were plotted to visualize the model's learning process and identify any potential overfitting or underfitting.

Outcome

The trained model achieved promising results with good accuracy on the validation set. The visualization showed that the model learned effectively without significant overfitting. This project showcases my skills in image processing, deep learning, and model optimization, demonstrating my ability to handle multi-class image classification tasks effectivel

git clone https://github.com/shabanaalina14/MRI_brain_cancer_classification/tree/main

Contact For any questions or suggestions, please feel free to reach.

Gmail: shabanaalina14@gmail.com

GitHub Profile: https://github.com/shabanaalina14

Happy Coding!

