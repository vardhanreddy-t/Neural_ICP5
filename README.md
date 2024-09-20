# Neural_ICP5
# Neural-Network-Assignment-ICP_5

NAME: PRIYA VARDHAN REDDY TAMMA ID#: 700764913

Video link:

In class programming:

1. Follow the instruction below and then report how the performance changed.(apply all at once)
• Convolutional input layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
• Dropout layer at 20%.
• Convolutional layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
• Max Pool layer with size 2×2.
• Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
• Dropout layer at 20%.
• Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
• Max Pool layer with size 2×2.
• Convolutional layer, 128 feature maps with a size of 3×3 and a rectifier activation function.
• Dropout layer at 20%.
• Convolutional layer,128 feature maps with a size of 3×3 and a rectifier activation function.
• Max Pool layer with size 2×2.
• Flatten layer.
• Dropout layer at 20%.
• Fully connected layer with 1024 units and a rectifier activation function.
• Dropout layer at 20%.
• Fully connected layer with 512 units and a rectifier activation function.
• Dropout layer at 20%.
• Fully connected output layer with 10 units and a Softmax activation function
Did the performance change?

Solution:

This code implements and evaluates two convolutional neural network (CNN) models using the CIFAR-10 dataset, a well-known benchmark in machine learning for image classification tasks. The original model consists of a simpler architecture with fewer layers and is trained for 5 epochs, while the modified model features a deeper architecture with additional convolutional and dropout layers, trained for 100 epochs. The dataset is first normalized and one-hot encoded for the classification task. After training, both models are evaluated on test data to determine their accuracy, and the performance change between the two models is calculated and displayed. This setup allows for a clear comparison of the impact of architectural complexity and training duration on model performance.

![image](https://github.com/user-attachments/assets/4ae12bd7-67d7-4a33-a4a0-5f1bd9e7682b)

![image](https://github.com/user-attachments/assets/660dd242-142f-4b6f-b4d2-ab461eded4d1)

![image](https://github.com/user-attachments/assets/15cd37bf-9a8e-4419-ad42-e168eeb99b3b)
        
![image](https://github.com/user-attachments/assets/c955efbd-2e21-480c-8595-a3043c59bf67)

2. Predict the first 4 images of the test data using the above model. Then, compare with the actual label for those 4 images to check whether or not the model has predicted correctly.

Solution:

This code segment visualizes the predictions made by the modified convolutional neural network (CNN) model on the CIFAR-10 test dataset. After training the model, it predicts the classes for the first four images in the test set and compares these predictions with the actual classes. The predicted and actual class labels are printed for review. Additionally, the segment includes a visualization of the first four test images, displaying each image along with its predicted and actual class labels in a grid format. This helps to intuitively assess the model's performance and identify any misclassifications.

![image](https://github.com/user-attachments/assets/398ef69d-0465-4ed4-96c3-7b962f8d9fcb)

3. Visualize Loss and Accuracy using the history object

Solution:

This code section fits the modified convolutional neural network (CNN) model on the CIFAR-10 training dataset and captures the training history, which includes metrics such as accuracy and loss over the specified number of epochs. After training, it generates two plots: the first visualizes the training and validation accuracy across epochs, allowing for an assessment of the model's performance on both training and test data. The second plot illustrates the training and validation loss, providing insights into the model's learning dynamics and potential overfitting. Together, these visualizations help in evaluating the effectiveness of the model and understanding how it improves with each epoch during training.

![image](https://github.com/user-attachments/assets/e923e87f-2809-4c68-86a6-99ae78a9eed1)

![image](https://github.com/user-attachments/assets/414aa0d6-0ba0-4ddf-9166-e77c803efbe3)
