# MNIST-Digit-Recognition-using-CNN
Handwritten digit recognition is a fundamental task in the field of computer vision, with numerous applications ranging from postal mail sorting to bank check processing. Deep learning techniques, particularly Convolutional Neural Networks (CNNs), have shown remarkable success in achieving high accuracy for this task. In this project, we aim to implement a CNN-based model using Tensorflow to recognize handwritten digits.

## The Logic Behind the Separation of Training Set and Test Set
Let's say you want to bake 20 cakes for a party. For each of the 20 cakes, you would need to prepare the ingredients and baking procedures. In order to do this, 20 lists of ingredients and 20 sets of baking directions, one for each cake, would need to be created.
Similar to this, each individual image in a dataset of photos, like the MNIST dataset, needs to have its data prepared. In order to do this, the photos must be divided into two sets: a training set and a test set.
You must have a label that identifies the right class (i.e., the digit represented in the image) for each image in the set of images.  This label tells the model what the appropriate output should be for a specific input, much like the set of baking instructions for a cake.
In terms of machine learning, the input data is frequently labeled as X and the associated labels as y. We would therefore have X_train and y_train for the training set and X_test and y_test for the test set in the case of the MNIST dataset.
Separating the data in this way allows us to train our model on a subset of the data (the training set), and then test its performance on a separate subset (the test set) to evaluate how well it can generalize to new, unseen data.
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

## The Logic Behind the Reshaping
Consider that you have a number of ingredients that you need to combine in order to bake a cake for a celebration. There is a set amount for each component, such as 2 cups of flour, 1 cup of sugar, etc. To produce the cake batter, you must combine all the components.
Now suppose you want to bake 20 cakes for the celebration. One cake's ingredients shouldn't be mixed at a time because that would take too long. As an alternative, you should combine all the ingredients at once to generate a large batch of cake batter, then divide that mixture into 20 equal portions to create 20 cakes.
In this illustration, the ingredients stand in for the MNIST images' pixels, the cakes for the actual MNIST images, and the batch of cake batter for the neural network's input data. The term "batch size" describes how many cakes (or photos) are processed all at once. By rearranging the photos to assume the shape (28, 28, 1), we are combining all of the component pixels into a single batch that the neural network can process all at once.
The "channels" refer to the different colors of the image. For example, if we had a color image of the cake, it might have three channels (red, green, and blue). However, the MNIST dataset consists of grayscale images, which only have one channel. So when we reshape the images to have a shape of (28, 28, 1), we are telling the neural network that each image only has one color channel.
```python
X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
```
## Data Preprocessing
Before we proceed further, there are a couple of important preprocessing steps that we need to perform on the MNIST dataset.

### Normalizing Pixel Values
Firstly, we need to normalize the pixel values of the images. The pixel values in the MNIST dataset range from 0 to 255, where 0 represents black and 255 represents white. However, it's common practice to scale the pixel values to be between 0 and 1. This scaling helps in improving the performance of the model during training. To achieve this, we will divide all the pixel values in the training set by 255. This ensures that all pixel values are within the range of 0 to 1, making them easier to work with.
```python
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
```
### One-Hot Encoding Labels
Secondly, we need to convert the labels in the dataset into categorical data. In the MNIST dataset, the labels represent the digits from 0 to 9. However, for training a multi-class classification model, it's beneficial to transform the labels into a one-hot encoded format. This means that each label is converted into a binary vector with a length of 10, where the index corresponding to the correct digit is set to 1 and all other indices are set to 0. This transformation allows the model to effectively learn the relationships between different digits and make accurate predictions.
The number of classes in the dataset is represented by the length 10 in the one-hot encoded vectors. The digits from 0 to 9 are represented by ten different classes in the MNIST dataset. As a result, in order to include all potential classes, we must generate binary vectors of length 10.
For example, if the label of a particular image is 3, the one-hot encoded vector representation would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]. The index corresponding to the digit 3 is set to 1, while all other indices are set to 0.
To perform these preprocessing steps, we will be using the to_categorical function from Keras to convert the labels into categorical data.
```python
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
```

## Building the CNN Model
Let's start with the structure. We will be using Sequential model. In Keras, the Sequential model is like a container that allows you to build a neural network layer by layer. You can think of it as a sequence of layers stacked on top of each other.
```python
model = Sequential()
```
### First Convolutional Layer
We will apply the Conv2D layer in our CNN. This layer is essential for detecting patterns and features in an image. It works by applying a set of small filters to the input image, where each filter specializes in detecting a specific pattern or feature.
Imagine you have a security camera that captures images of people entering a building. You want to use a neural network to automatically detect if someone is wearing a hat or not.
Here, the Conv2D layer performs the function of a filter by scanning the collected images and searching for hat-like patterns. Pixel by pixel, it moves over the image, examining discrete areas at a time. It looks for certain characteristics that mimic hats in each part, such as a round shape on top of a person's head.
The Conv2D layer may learn to detect a variety of hat-related patterns, such as varied shapes, sizes, and orientations, by applying numerous filters with unique properties.
A feature map that highlights the portions of the image with hat-like patterns is the result of the Conv2D layer. The neural network's further layers then receive this feature map for additional processing and decision-making.
So, in this example, the Conv2D layer acts as a "hat detector," scanning the image for specific patterns associated with hats and helping the neural network identify whether a person is wearing a hat or not.
The Conv2D layer will apply these filters to the image and carry out a convolutional mathematical technique.  Convolution is the process of multiplying the filter values by the matching image pixel values and adding them together. Each area that the filter covers in the image is subjected to the same process.
The Conv2D layer generates a new feature map that highlights the regions of the image where the patterns or features were found by using this convolution technique. High values on the feature map represent these regions, whereas the rest of the image is mainly unchanged. 
In our case, we will apply a Conv2D layer with 32 filters, each having a size of 3x3. We will use the 'relu' activation function, which helps introduce non-linearity to the model. The input shape of our images is (28, 28, 1), indicating that they are 28x28 pixel grayscale images.
32 filters imply that there will be 32 separate filters in the Conv2D layer, each of which can identify a certain characteristic. The model can learn a broader range of features and recognize more intricate patterns in the data when there are more filters available.
The quantity of the dataset and the problem's complexity both affect how many filters are used. The ability of the model to learn complex details in the data may be enhanced by using more filters. However, it also makes the model's computations more difficult and memory-intensive.
In real-world applications, experimentation and fine-tuning are frequently employed to establish the amount of filters to be utilized in a Conv2D layer. It's typical to start with fewer filters and gradually raise the number if more are required, depending on the model's performance and the difficulty of the task at hand.
```python
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
```

#### Note for RELU function
Activation functions in neural networks are essential for adding non-linearity to the model. The network can learn intricate patterns and more complex predictions thanks to non-linearity.
One of the most often used activation functions in deep learning models, including CNNs, is the'relu' activation function, which stands for Rectified Linear Unit. It functions by maintaining positive values and setting negative values to 0. It has the following mathematical representation:
```python
relu(x) = max(0, x)
```
Here, 'x' represents the input value to the activation function. If 'x' is positive, the 'relu' function returns the same value, and if 'x' is negative, it outputs zero.
Consider a neuron that receives a value of -2 as an input. The'relu' activation function causes the neuron to output 0 when the input is negative. In this instance, the neuron is effectively signaling that the input is not important enough to cause activation or signal transmission.
Let's now think about an optimistic input value of 3. Because it is a positive number, the'relu' activation function will return the same result, which is 3. In other words, the neuron is engaged and transmits the positive input value exactly as it is.
To put it simply, the'relu' activation function functions as a switch that, when activated for positive input values, allows information to pass through while being blocked when activated for negative input values. The network can recognize and learn complicated links and patterns in the data thanks to its non-linear nature.
In our CNN model, we enable the neurons to selectively activate based on the positive or negative values of the inputs, and this helps the model recognize significant features and patterns in the images. We do this by employing the'relu' activation function in the Conv2D layer.

### First MaxPooling Layer
Then we will apply MaxPooling2D layer. The MaxPooling2D layer can be thought of as a means to condense the data in the feature maps. The feature maps are divided into manageable portions, and only the highest value from each section is retained. By doing this, it is possible to shrink the feature maps without losing any crucial data.
Consider searching for the largest integer within each discrete area of a grid of numbers. You would make pools (small groups) of numbers and choose the largest number from each pool. You end up with a smaller grid of the largest numbers from each pool by doing this.
In our CNN model, the MaxPooling2D layer does something similar. It takes the feature maps generated by the previous Conv2D layer and divides them into small pools. For each pool, it selects the maximum value and keeps only that value. This helps to reduce the size of the feature maps while retaining the most important features.
Our model uses less memory and computes more quickly thanks to the MaxPooling2D layer. Additionally, it aids the model's attention on the image's most striking details, improving its capacity to spot patterns and things.
```python
model.add(MaxPooling2D(pool_size=(2, 2)))
```

### Second Conv2D Layer
We give the model the ability to learn deeper and more complicated features by introducing another Conv2D layer. The second Conv2D layer is often used to learn higher-level features (more complicated patterns, forms, facial features, etc.), whereas the first Conv2D layer is typically used to learn simpler features (such as edges, corners, etc.).
The reason for using 64 filters in the second Conv2D layer is to have a greater capacity to learn more features. By using more filters, the model can explore a larger number of feature combinations and capture more complex relationships. This allows the model to have a deeper feature learning capability, which generally leads to better overall performance.
This results in the formation of a feature extraction hierarchy where the model gradually learns more complicated features and creates a more useful representation. This could improve the model's classification performance.
```python
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
```

We may further downsample the feature maps, minimize the spatial dimensions, and concentrate on the most crucial features while also increasing computing efficiency by adding the second MaxPooling2D layer.
```python
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
```

## The Journey So Far: Unveiling the Mysteries of Image Recognition with a Forest Photo Example (1/2)
 To grasp what we have done better, let's embark on a journey through an image of a nighttime forest with trees, the moon, and wolves:
 
**Step 1: First Convolutional Layer (Conv2D)**
First, we take our image, which represents a nighttime forest scene. Each pixel in the image contains a color value.
The filters in our first convolutional layer help us recognize different features present in the forest. For example, one filter may recognize bears, another filter may recognize tree trunks, and another filter may recognize stars or the moon.
These filters scan the image and become sensitive to different features, creating various feature maps.  

**Step 2: First Pooling Layer (MaxPooling2D)**
We take the feature maps obtained from the convolutional layer. These feature maps may contain features such as silhouettes of trees, the shape of the moon, or the contours of wolves.
The pooling layer reduces the size of these feature maps while preserving important details. For example, it can reduce the details of trees, sharpness of the moon, or fine lines of the wolves while decreasing their dimensions.
Let's say we want to recognize the shape of a wolf in that image. The image could be large in terms of height and width. The max pooling operation helps us reduce this image to a smaller size while preserving important features.
First, we place a window over each region of the image. This window has a specific size, for example, 2x2 pixels. Then, we create a new window by taking the maximum value of the pixels in each window.
When we apply this operation to the entire image, the size of the image shrinks, and we obtain a more general representation. For instance, to recognize the shape of the wolf, we don't need all the fine details present in the original image. By using the max pooling operation to reduce the image size, we can preserve the overall contours and important features of the wolf.
This way, we obtain a more general representation. It enables recognition of higher-level features and the ability to recognize similar features in different forest scenes.

**Step 3: Second Convolutional Layer (Conv2D)**
In the second convolutional layer, we use more filters to recognize more complex features. For example, we add filters to recognize the branches and leaves of trees, capturing finer details.
In this layer, we further process the feature maps from the previous layer to obtain more abstract feature maps.

**Step 4: Second Pooling Layer (MaxPooling2D)**
The second pooling layer further reduces the size of the feature maps from the previous layer.
This highlights important features in the forest, such as the overall shape of trees, the size of the moon, or the general contours of wolves, while reducing their dimensions.

### Fully Connected Layer
In the context of 2D feature maps, each pixel represents a point in the feature map. The feature maps have two axes: the horizontal axis (usually referred to as the x-axis) and the vertical axis (usually referred to as the y-axis).

Each pixel corresponds to a specific location and has a value associated with it. This value represents the feature or pattern present at that particular location. For example, when an edge-detecting filter is applied, the pixel values may vary depending on the presence of an edge.

These pixel values allow for the representation of the feature maps in the form of matrices. Each cell in the matrix contains the value of a pixel. This matrix representation is used to better visualize and process the features in the image.

Therefore, you can think of each axis in the 2D feature maps as representing the position of each pixel. These axes help organize the features in a structured matrix format.
In a convolutional neural network (CNN), the convolutional layers and pooling layers generate 2D feature maps that capture spatial information and learned features from the input data. However, when we want to pass these feature maps to a fully connected layer (also known as a dense layer), we need to convert the 2D representation into a 1D vector.

The Flatten() layer performs this conversion by reshaping the 2D feature maps into a linear sequence. It takes the grid-like structure of the feature maps and transforms it into a single row, effectively "flattening" the data.

By flattening the feature maps, we preserve the spatial relationships between the elements in the maps while converting the data into a format that can be easily processed by the fully connected layers. This allows the subsequent layers to receive the feature information in a sequential manner and perform computations on the flattened vector.

In essence, the Flatten() layer acts as a bridge between the convolutional and fully connected layers, enabling the neural network to utilize the spatial information captured by the convolutional layers while operating on a 1D representation of the features.

So, the Flatten() layer in this context transforms the 2D feature maps into a 1D vector, preparing them for further processing by the fully connected layers in the neural network.

```Flatten the feature maps
model.add(Flatten())
```

The fully connected layer in a convolutional neural network (CNN) is a sort of layer that links every neuron in the preceding layer to every neuron in the following layer. As a result of its extensive connection, it is sometimes referred to as a dense layer.
To our CNN model, we are at this point adding a fully connected layer. This layer is represented by the Dense function. There are 128 neurons or units in this layer, as indicated by the number. Depending on the task's difficulty and requirements, this value can be changed.
We are adding more learnable parameters to the model by adding this fully connected layer. This layer's neurons will take inputs from the layer below and compute a weighted sum of those inputs before performing the activation function. This layer's outputs will serve as inputs for the model's future layers.
```Add a fully connected layer
model.add(Dense(128, activation='relu'))
```

The neural network's final layer, the output layer, is often employed for classification or prediction tasks. Based on the features from the preceding layer, this layer computes the results.
Using the Dense function, we are adding the output layer in this phase. The output layer's number of neurons or units is indicated by the number 10. The number of classes we wish to predict affects this value. For instance, we would set it to 10 if we wanted to forecast 10 different classes.
The parameter activation='softmax' specifies the activation function for this layer. The softmax activation function is commonly used in classification problems. It generates probability values for each class, and the sum of these probabilities is equal to 1. This allows the model to produce a probability distribution for each class and make predictions based on the highest probability.
The neural network is equipped to generate predictions thanks to this output layer. Based on the input features, classification is done, and the probabilities between the classes are returned.
In this manner, the output layer enables classification and prediction tasks for the model during training.
```Add the output layer
model.add(Dense(10, activation='softmax'))
```

### Compiling the model
The model is being put together in the last step, which entails setting up the neural network's learning process. During the training stage, a loss function, an optimizer, and evaluation metrics will be applied.
loss = "categorical_crossentropy" It involves choosing a method to assess the model's performance during training. In particular, it describes the loss function that is used to compute the discrepancy between the model's predicted output and the actual expected output.

We are use the categorical cross-entropy loss function in this instance. When dealing with classification issues where there are numerous classes to choose from, this loss function is frequently used. It compares the expected probability of each class with the actual probabilities (one-hot encoded labels), and then it determines a single number to indicate the overall difference or error between the two.
Larger discrepancies between projected and actual probabilities will be penalized more harshly by the categorical cross-entropy loss function, which will encourage the model to make better predictions over time. The model seeks to improve the accuracy of accurately classifying the input data into each of its classes by minimizing this loss function.
It involves choosing the optimization algorithm that will be applied to change the neural network's weights during training. The optimizer is essential in modifying the model's parameters to reduce loss and boost efficiency.

Adam, which stands for Adaptive Moment Estimation, is the optimizer that we are employing in this situation. Adam is a popular optimizer algorithm that combines the advantages of RMSprop and AdaGrad, two additional optimization techniques. It is renowned for its effectiveness, quickness, and strong performance across a range of jobs.
The gradients of the loss function are used to calculate and update the weights in the Adam optimizer. According to the first and second moments of the gradients, it maintains adaptive learning rates for each weight and modifies them separately. The optimizer can converge more quickly and effectively handle various types of data and designs because to its variable learning rate.

To help the model produce better predictions and perform better overall during training, we use the Adam optimizer to identify the ideal set of weights that minimizes the loss function.
The evaluation metric(s) that will be calculated during training and evaluation are specified by metrics=['accuracy']. The accuracy metric, which measures the proportion of samples that are properly predicted, is what we are utilizing in this instance. It gives a hint as to how accurately the model is classifying data, i.e., how well it is functioning.

```Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## The Journey So Far: Unveiling the Mysteries of Image Recognition with a Forest Photo Example (2/2)
**Step 1: Flattening the Feature Maps**
In the previous steps, we processed our image containing the forest, trees, and wolf and obtained feature maps. In this step, we flatten these feature maps. Instead of considering each pixel as a separate feature, we combine all the features into a single vector.

**Step 2: Adding a Fully Connected Layer**
Taking the flattened feature vector, we add a fully connected layer. This layer allows for higher-level processing of the features. In this case, we add a fully connected layer with 128 neurons. The neurons are activated using the ReLU activation function, aiming to capture more complex combinations of features.

**Step 3: Adding the Output Layer**
Finally, we add the output layer. This layer enables the model to make predictions for different classes, such as forest, trees, and wolf. If we have, for example, 10 different classes, this layer would have 10 neurons. These neurons are normalized using the softmax activation function, providing probabilities for each class.

**Step 4: Compiling the Model**
Lastly, we compile the model. During the compilation stage, we specify how the model should be trained and how its performance should be measured. We use 'categorical_crossentropy' as the loss function since we are dealing with a multi-class classification problem. The 'adam' optimizer is chosen as it is an efficient and effective optimization algorithm. The performance metric used is 'accuracy', which allows us to track the percentage of correct classifications made by the model.

### Evaluation of the Model
```
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
```