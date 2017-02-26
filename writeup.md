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


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![output of random test image][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/image_plot.png]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I chose not to do any image preprocessing.  I wanted to test out if the use of any preprocessing steps was useful in gernalizing the images.  I could have converted the images to grayscale, which would have helped gernalize the images, however I chose leave the images as 3 dimensional RGB images for the additional spectral dimensions.  In additional, I could have used a gaussian filter.  The filter would remove noise and also hlep gernalize the images.  In this case, the images were small so I chose not to use a filter because I may lose some details.  Overall, the accuracy did not improve significally with these steps, therefore to save on some computation time I left these pre-processing steps out. 

![alt text][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/image_plot.png]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn.model_selection.train_test_split fucntion and randomly extracted 20% of the test data set for the validation data set.

The original training data set had 31367 images, I agumented my data set with rotatated images. My final training set had 62734 number of images. My validation set and test set had 7842 and 12630 number of images.

The thrid code cell of the IPython notebook contains the code for augmenting the data set. I augmented my dataset with rotated images.  I took the test set of images and rotated them by 25 degrees using rotate from the sklearn.tranform module.  I created a new numpy array for the rotated images and labels, afterwhich I concatenated the rotated array to the test array and label.  Adding to the dataset would help the neural network optimize on more images.

Here is an example of an original image and a rotated augmented image:

![image original and rotated][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/rotate.png]

The difference between the original data set and the augmented data set is the augmented image is rotated by 25 degrees. The training set essentially doubled from 31367 to 62734.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| l2 normalization		| etc.        									|
| Dropout				| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used a base LeNet CNN. 
sigma
Epoch
batch size
learning rate
AdamOptimizer
The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients
an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.971
* validation set accuracy of 0.972 
* test set accuracy of 0.912

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Bumpy road - 22][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/1.jpg] ![Yield - 13][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/2.jpg] ![No passing - 9][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/3.jpg] 
![Road work - 25][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/4.jpg] ![Speed limit (30km/h) - 1][https://github.com/trevordo/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/5.jpg]

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

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .99         			| Road narrows on the right							| 
| 1.00    				| Yield 											|
| .81					| End of no passing by vehicles over 3.5 metric tons|
| .99	      			| Right-of-way at the next intersection				|
| .53				    | Speed limit (30km/h) 								|


For the second image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| 1.00    				| Yield 											| 
| 1.00    				| Yield 											|
| .81					| End of no passing by vehicles over 3.5 metric tons|
| .99	      			| Right-of-way at the next intersection				|
| .53				    | Speed limit (30km/h) 								|

For the thrid image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .81					| End of no passing by vehicles over 3.5 metric tons| 
| 1.00    				| Yield 											|
| .81					| End of no passing by vehicles over 3.5 metric tons|
| .99	      			| Right-of-way at the next intersection				|
| .53				    | Speed limit (30km/h) 								|

For the fourth image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .99	      			| Right-of-way at the next intersection				| 
| 1.00    				| Yield 											|
| .81					| End of no passing by vehicles over 3.5 metric tons|
| .99	      			| Right-of-way at the next intersection				|
| .53				    | Speed limit (30km/h) 								|

For the fifth image ...

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .53				    | Speed limit (30km/h) 								| 
| 1.00    				| Yield 											|
| .81					| End of no passing by vehicles over 3.5 metric tons|
| .99	      			| Right-of-way at the next intersection				|
| .53				    | Speed limit (30km/h) 								|