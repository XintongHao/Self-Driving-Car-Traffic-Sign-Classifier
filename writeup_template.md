# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_img/training_sample.png "training_sample"
[image2]: ./output_img/histogram_classes.png "histogram_classes"
[image3]: ./output_img/normalized_img.png "normalized_img.png"
[image4]: ./output_img/test_sample.png "test_sample"
[image5]: ./output_img/prediction_sample.png "prediction_sample"
[image6]: ./output_img/probabilities.png "probabilities"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

This is three random samples from training set with their sign names.

![alt text][image1]

This the histogram of classes distribution.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will use less parameters to do the computations than color images which have 3 channels.

Then, I normalized the image data to make the data has mean zero and equal variance. Because normalization will make the model to compare the results easier and speed up the process.

Here is an example of an original image and output image:

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 1x1x400								|
| RELU					|												|
| Fully connected		| outputs 400 
| Fully connected		| outputs 120 
| Fully connected		| outputs 84 
| Fully connected		| outputs 43 


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used `tf.nn.softmax_cross_entropy_with_logits` function to compare the logits to the true labels and calculate the cross entropy. Then use `tf.reduce_mean` function to average the cross entropy from all of the training images. I also used `AdamOptimizer` to use Adam algorithm with learning rate of `0.0009` to minimize the loss function similarly to SGD does.
In the traning process, I set the `EPOCHS = 60` and `BATCH_SIZE = 128`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 99.2%
* validation set accuracy of 93.6%
* test set accuracy of 92.6%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
My model is based on LeNet network architecture which has a high performance on image classification. From previous classes, I learnt that this network was good at classify MNIST dataset, then I tried to use it to classify the traffic signs and the results were good as well.
First, I used LeNet basic network architecuture to train the model, the highest accuracy is around 89%. Then I tried to add one more 1x1 convolutions, which was mentioned in the lecture. After three convolutional layers, the output dimension is 1x1x400. Then it follows three fully connected layers and output 43-class logits. The result showed that the changes made the model better which had 93% validation accuracy


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4]

The first and third images shoud be the easiest images recognize which have clear outline and less noise on the signs. The second and fifth images don't have clear outline and need the model to tell the direction. The fourth and sixth images were compressed and were out of the shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image5]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (60km/h) 									| 
| No left turn		| Speed limit (70km/h)									|
| Stop					| Stop											|
| Yield     		| Yield				 				|
| Go straight or left			| Go straight or left    							|
| No entry			| No entry 							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. This compares favorably to the accuracy on the test set of 92.6%. It could recognize the shape of the Speed limit sign but has difficulty to read the number on the sgin. It also made mistake on the No left turn sign. I think it may view the left turn as "7". 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here is the visualization of the sofmax probabilities:

![alt text][image6]

Unfortunately, for the first two wrong prediction, the model didn't get the right answer in the top 3 probabilities. But I found the model have probelm on distinguish "left turn" and "7". To make it perform better, I can do up-sampling on sign which have "left turn" or "7".  

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


