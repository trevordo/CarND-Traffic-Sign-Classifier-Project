#**Traffic Sign Recognition** 

##Writeup Template

###Template written by andrewpaster.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 31367
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the initial data.

![output of random test image](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/chart.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I chose not to do any image preprocessing.  I wanted to test out if the use of any preprocessing steps was useful in gernalizing the images.  I could have converted the images to grayscale, which would have helped gernalize the images, however I chose leave the images as 3 dimensional RGB images for the additional spectral dimensions.  In additional, I could have used a gaussian filter.  The filter would remove noise and also hlep gernalize the images.  In this case, the images were small so I chose not to use a filter because I may lose some details.  Overall, the accuracy did not improve significally with these steps, therefore to save on some computation time I left these pre-processing steps out. 

![image data plot](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/image_plot.png)

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn.model_selection.train_test_split fucntion and randomly extracted 20% of the test data set for the validation data set.

The original training data set had 31367 images, I agumented my data set with rotatated images. My final training set had 62734 number of images. My validation set and test set had 7842 and 12630 number of images.

The thrid code cell of the IPython notebook contains the code for augmenting the data set. I augmented my dataset with rotated images.  I took the test set of images and rotated them by 25 degrees using rotate from the sklearn.tranform module.  I created a new numpy array for the rotated images and labels, afterwhich I concatenated the rotated array to the test array and label.  Adding to the dataset would help the neural network optimize on more images.

Here is an example of an original image and a rotated augmented image:

![image original and rotated](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/rotate.png)

The difference between the original data set and the augmented data set is the augmented image is rotated by 25 degrees. The training set essentially doubled from 31367 to 62734.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6	|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16		|
| Fully connected		| flatten, Output = 400							|
| l2 normalization		| dimension 0, epsilon=1e-12					|
| Fully connected		| Output = 120 									|
| RELU					|												|
| Fully connected		| Output = 84  									|
| RELU					|												|
| Dropout				| keep probability 0.8							|
| Fully connected		| Output = 43  									|
| Softmax				| 												|
|						|												|

 
####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used a base LeNet CNN. This CNN is the basis of the previous lesson.  I added dropout to prevent overfitting, here I used a high keep probability parameter. A l2 normalization was added after flatten to add some generalizability to the model. Sigma and mu hyperparameter were present in the LeNet lesson and applies to the generation of weights.  Epoch and batch size can be found in cell 8 and this is were we assign the number of iterative cycles and the size of each batch to be fed into the LeNet CNN, respectively.  The learning rate hyperparameter can be found in cell 11 along with the dropout hyperparameter.  The learning rate is applicable to the rate of movement in your opitmizer, AdamOptimizer is this case, will apply to your model.  Gernerally, there is no relationship with learning and the accuracy, a higher (faster) learning rate will not achieve higher accuracy and vicevera.  If your error is experiencing wild swings then it's a good idea to lower the rate.  The optimizer used is an AdamOptimizer and this function helps us run a stochastic gradient decent and backpropagate and update weights for a forward pass. The AdamOptimizer is described as computation efficient and is ideal for large data sets.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.971
* validation set accuracy of 0.972 
* test set accuracy of 0.912

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture that was part of the lesson plan for the MNIST tutorial.  This architure worked well for the that data set so I decide to modify and try a few things like drop out and l2 normalization.

* What were some problems with the initial architecture?
The initial architecture was for optimized for 10 classes and 1 dimension grayscale images.  The sigma for the 

* How was the architecture adjusted and why was it adjusted? 
I first added a dropout to prevent overfitting, and l2 normalization for additional generalization.  I changed the sigma from 0.1 to 0.2 for additional weight change.  Additional adjustment can include more layers and different activation function.

* Which parameters were tuned? How were they adjusted and why?


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
LeNet was chosen.

* Why did you believe it would be relevant to the traffic sign application?


* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The training, validation and test set was all reporting >0.90 accuracy. This model predicted 9/10 images correctly, they all fell in the same range.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Bumpy road - 22](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/1.jpg) ![Yield - 13](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/2.jpg) ![No passing - 9](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/3.jpg) 
![Road work - 25](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/4.jpg) ![Speed limit (30km/h) - 1](https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/5.jpg)

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| Bumpy road      		| Road narrows on the right							| 
| Yield     			| Yield 											|
| No passing			| End of no passing by vehicles over 3.5 metric tons|
| Road work	      		| Right-of-way at the next intersection				|
| Speed limit (30km/h)	| Speed limit (30km/h) 								|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares unfavorably to the accuracy on the test set of 0.971

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook. 

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Probabilities

[[ 0.9999907   0.00000536  0.00000199  0.00000162  0.0000002 ]
 [ 1.          0.          0.          0.          0.        ]
 [ 0.8180477   0.18047003  0.0006249   0.0003275   0.00028416]
 [ 0.99096     0.00709494  0.00190972  0.00003307  0.00000226]
 [ 0.5364337   0.33381796  0.12948893  0.00022669  0.00002666]]
 
 Predictions
 
 [[24 21 22 26 18]
 [13 17 14 12 21]
 [42  1 28 41 32]
 [11 30 42 23 12]
 [ 1  0  4  5 35]]
 

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .99         			| Road narrows on the right							| 
| .00    				| Double curve										|
| .00					| Bumpy road										|
| .00	      			| Traffic signals									|
| .00				    | General caution 									|
 
For the second image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| 1.00    				| Yield 											| 
| .00    				| No entry 											|
| .00					| Stop												|
| .00	      			| Priority Road										|
| .00				    | Double curve 										|

For the thrid image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .81					| End of no passing by vehicles over 3.5 metric tons| 
| .18    				| Speed limit (20km/h) 								|
| .00					| Children crossing									|
| .00	      			| End of no passing 								|
| .00				    | End of all speed and passing limits 				|

For the fourth image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .99	      			| Right-of-way at the next intersection				| 
| .01    				| Beware of ice/snow 								|
| .00					| End of no passing by vehicles over 3.5 metric tons|
| .00	      			| lippery road										|
| .00				    | Priority road 									|

For the fifth image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .53				    | Speed limit (30km/h) 								| 
| .33    				| Speed limit (20km/h) 								|
| .12					| Speed limit (70km/h)								|
| .00	      			| Speed limit (80km/h)								|
| .00				    | Ahead only 										|