---
layout: post
title: Beginner's guide to Understanding Convolutional Neural Networks
published: true
comments: true
permalink: /learningblog/
category: learning
---

_This post is written by summarizing few blog post that explained Convolutional Neural Networks with adding some of my ideas._

![LeNet](/images/LeNet.png)  
_Picture shows the basic CNN model that proposed by Yann LeCun in 1998, after that, all the CNN is built based on it._  
  
    
3 good links that can let beginner to understand CNN easily:

* [Youtube video to have a clear picture of CNN by Brandon Rohrer](https://www.youtube.com/watch?v=FmpDIaiMIeA&t=870s)
* [Intro to CNN by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
* [Deeper understanding on CNN by Ujjwal Karn](https://www.kdnuggets.com/2016/11/intuitive-explanation-convolutional-neural-networks.html)

More intense learning on CNN(with code explained):
* [CS231n by Stanford University](http://cs231n.github.io/convolutional-networks/#conv)  

Playing around with the filter to get the concept of CNN:
* [Image Kernel by Victor Powell](http://setosa.io/ev/image-kernels/)  

Course I highly recommended:  
* [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)

> ConvNets are good at finding patterns, which is to capture local "spatial"(things that next to one another) patterns of data, so it's  best to apply on image because the patterns(Pattern in the images mean that the position of the data matters to us)in the image are the  most common one, and if the data fails to be made to look like an image, CNN will be less useful.  
**Depends on the problem and decide to which architecture of the network to be used on the problems.**

## Convolutional Neural Networks is separated into 2 parts:
* **Feature learning**
* **Classification**

> _Images are a matrix of pixel values._  
[**Channel**](https://en.wikipedia.org/wiki/Channel_(digital_image)) is a conventional term used to refer to a certain component of an image. An image from a standard digital camera will have three channels – red, green and blue(RGB) – you can imagine those as three 2d-matrices stacked over each other (one for each color), each having pixel values in the range 0 to 255.  

> While a grayscale image, has just one channel. The value of each pixel in the matrix will range from 0 to 255 – zero indicating black and 255 indicating white.  

****
## Feature Learning
### 1. Convolution Part
![Example of image](/images/convimage.png)  
_Example of 5x5 image using only 1 and 0 pixel to illustrate_  
![Example of filter](/images/convfilter.png)  
_Example of 3x3 filter using only 1 and 0 pixel to illustrate_  
> Think of this 2 image as a matrix with number, 0 act as white and 1 act as black.

At this point, this filter is like a "X" detector for the image, which will act like a sliding window(animation down there), to search the "X" pattern in the image. If want to detect another pattern, will use another type of filter to slide around the images, here is where the depth of next layer come in, let's say you use 3 filter, which is "X", "O" and "I" detector to slide through the images, then the next layer of the network will be depth=3. (Will explain more down there.)  
The matrix can also be the color contrast, texture, etc. of the images.  

![Convolution animation](/images/convolution.gif)  
Convolved feature also called feature maps/activation map.
This is the process of convolving, which take the **filter** to do element-wise multiplication with the matrix pixel of the image. After convolving, the convolved feature which is the output of the previous layer, will be the input for the next layer. If the pixel in the image match the pixel of the filter, the pixel in convolved feature will be very high, if there's nothing related to the **filter** in the image, then the pixel will be very low in the convolved feature, which is the next layer.  

> _Time takes to run the filter = ((height*width)of image)/stride_  

Good example to illustrate the working of filter on image.  
![Filter on image](/images/filteronimage.png)  
As you can see, there are few types of filter, which to scan through the image and convolve to the next layer, which with depth(equal to the number of filters used).  

![Convolution](/images/convolve.gif)  
 Great animation to explain how the feature detector affect the output and depth of output in the feature map.  
 For example the first edge detector is "\", it will output the image where the most obvious part of the image will be the part contained edges "\". Same goes to the second edge detector "/".  
 
 ### Parameters that control the behavior of each convolved layer
 * **Stride**
 * **Padding(Zero-padding)**
 * **Number of filter(depth of the layers)**
 * **Size of the filter**

****
 #### Stride
 Stride is the amount of filter shift in the image. The bigger the stride, the smaller the feature map.  
 
 ![Stride1](/images/Stride.png)  
 Stride = 1.  
 ![Stride2](/images/Stride2.png)  
 Stride = 2.
 
 
#### Padding(Zero-padding)
In the early stage, we want to preserve as much information of an image as possible, so we can extract low level features.  
Padding can preserve the dimension of the imageas well, or in another name - "same convoution".(Example in the image)  
![Padding](/images/padding.png)  
> Adding zero-padding is also called wide convolution, and not using zero-padding would be a narrow convolution.  
Formula:  
* Same convolution 
  -Zero padding = (Filter size - 1)/2 (Get the output (n+2p-f+1) by setting zero padding to be p=(f−1)/2 when the stride is 1 ensures that the input volume and output volume will have the same size spatially.)  

* Output(height/width) = ((Input(height/width)-filterSize+(2*zero-padding))/stride) + 1

> **Constraints on strides.** Note again that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size W=10, no zero-padding is used P=0, and the filter size is F=3, then it would be impossible to use stride S=2, since (W−F+2P)/S+1=(10−3+0)/2+1=4.5, i.e. **not an integer, indicating that the neurons don’t “fit” neatly and symmetrically across the input.**  
Therefore, this setting of the hyperparameters is considered to be **invalid**, and a ConvNet library could throw an exception or zero pad the rest to make it fit, or crop the input to make it fit, or something.  
As we will see in the ConvNet architectures section, sizing the ConvNets appropriately so that all the dimensions “work out” can be a real headache, which the use of zero-padding and some design guidelines will significantly alleviate.  
Real-world example. **The Krizhevsky et al. architecture** that won the _ImageNet challenge in 2012_ accepted images of size [227x227x3]. On the first Convolutional Layer, it used neurons with receptive field size F=11, stride S=4 and no zero padding P=0. Since (227 - 11)/4 + 1 = 55, and since the Conv layer had a depth of K=96, the Conv layer output volume had size [55x55x96]. Each of the 55*55*96 neurons in this volume was connected to a region of size [11x11x3] in the input volume. Moreover, all 96 neurons in each depth column are connected to the same [11x11x3] region of the input, but of course with different weights. As a fun aside, if you read the actual paper it claims that the input images were 224x224, which is surely incorrect because (224 - 11)/4 + 1 is quite clearly not an integer. This has confused many people in the history of ConvNets and little is known about what happened. _My own best guess is that Alex used zero-padding of 3 extra pixels that he does not mention in the paper._(From CS231n, Stanford University)

> According to Andrew Ng, if it's not an integer, we can use floor() function to round it down.  

#### Number of filter(depth of the layers)
_Example: Image: 6x6x3, filter: 3x3x3_  
Channel of filter must be same as the channel of the image, after convolving, will get 4x4xn, n is depends on the number of filter you use, in another words, means that depends on the number of feature detector you use.  

#### Size of the filter
Size of the filter usually is odd number so that the filter has the "central pixel"/"central vision" so to know the position of the  filter. As if f was even, then you need some asymmetric padding, or it's only f is not that this type of same convolution gives a natural padding. We can pad the same dimension all around them, pad more on the left and pad less on the right or something asymmetric.
 
### 2. Introducing Non-Linearity, ReLU (Rectified Linear Units) to the Layers
 > It could be other activation function, but by far ReLU works the best.  
The ReLU layer applies the function f(x) = max(0, x) to all of the values in the input volume. 
In basic terms, this layer just changes all the negative activations to 0.  
![ReLU](/images/ReLU.png)  
_The reason that we apply non-linearities to the function is that Convolution is a linear operation – element wise matrix multiplication and addition, so we account for non-linearity by introducing a non-linear function like ReLU, prevent from computing the linear function, which will be a bad model._  


ReLU helps in solving the vanishing gradient problem, which is a problem when we train the neural network using gradient-based algorithm, like sigmoid, it will squish all the gradient value into 0-1, then when performing the gradient descent, the gradient will be updated very small each time, and the time will take longer to complete.(Learning becomes slow.)[ReLU helps in solving this problem.] (https://www.quora.com/What-is-the-vanishing-gradient-problem)  
![ReLUimage](/images/ReLUimage.png)
Other non linear functions such as tanh or sigmoid can also be used instead of ReLU, but ReLU has been found to perform better in most situations.  
  
### 3. The Pooling Layer  
Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map and retains the most important information of an image. Spatial Pooling can be of different types: Max, Average, Sum etc.  
  
Instead of taking the largest element we could also take the average (Average Pooling) of sum of all elements in that window. In practice, Max Pooling has been shown to work better.  
  
Max pooling being the most popular. This basically takes a filter (example: size 2x2) and a stride of the same length(which is 2). It then applies it to the input volume and outputs the maximum number in every sub-region that the filter convolves around. The output size formula for the max pooling is same as the convolution one.  
  
Max pooling usually doesn't use any padding.  
**Only hyperparameter here, there's no parameters to learn here.**  
![Maxpooling](/images/maxpooling.png)  
Maxpooling in 2D image.  
![3dMaxpooling](/images/3dmaxpooling.png)  
Maxpooling in 3D image, which is the one we deal with in real world.  

![Maxpoolingbnw](/images/maxpoolingbnw.png)  
The function of Pooling is to progressively reduce the spatial size(the length and the width change but not the depth) of the input representation.  
In particular, pooling:  
 * Makes the input representations (feature dimension) smaller and more manageable.
 * Reduces the number of parameters and computations in the network, therefore, controlling overfitting. 
 * Makes the network invariant to small transformations, distortions and translations in the input image (a small distortion in input will not change the output of Pooling – since we take the maximum/average value in a local neighborhood).
 * Helps us arrive at an almost scale invariant representation of our image (the exact term is “equivariant”). This is very powerful since we can detect objects in an image no matter where they are located(or no matter how they rotate in the images). [This link explained very well.](https://www.quora.com/How-is-a-convolutional-neural-network-able-to-learn-invariant-features)  
 
### 4. Dropout Layer 
_(Not in the traditional architecture of CNN but very useful, because helps a lot in fighting overfitting)_  
  
The idea of dropout is simplistic in nature. This layer “drops out” a random set of activations in that layer by setting them to zero. Simple as that. Now, what are the benefits of such a simple and seemingly unnecessary and counterintuitive process? Well, in a way, it forces the network to be redundant. By that I mean the network should be able to provide the right classification or output for a specific example even if some of the activations are dropped out. It makes sure that the network isn’t getting too “fitted” to the training data and thus helps alleviate the overfitting problem.  
An important note is that this layer is only used during training, and not during test time.  

### 5. Network in Network Layer (1X1 convolution)  
_(Not in the traditional architecture of CNN but very useful, because helps in generating more features, and making network deeper in a computational inexpensive way.)_  

A network in network layer refers to a conv layer where a 1 x 1 size filter is used.   
1x1 convolutions span a certain depth, so we can think of it as a 1 x 1 x N convolution where N is the number of filters applied in the layer. Effectively, this layer is performing a N-D element-wise multiplication where N is the depth of the input volume into the layer.  
  
Example:  
If the previous layer has 128 feature maps (say) then "1x1 convolutions" are convolutions across all these feature maps with filters each of size 1x1x128. Say one chooses to have 64 of these 1x1x128 dim filters, then the result will be 64 features maps, each the same size as before. View each output feature map as "per-pixel" projections (dot-product) onto a lower dimensional space using a single learned filter (weights tied) across all feature maps. Basically, they just crush 128 feature maps (representational responses to 128 learned filters) into 64 feature maps ignoring the spatial dimension.  
  
Remember that larger filters like a 3x3x128 filter would also learn to summarize feature responses across all feature maps so in this way all size filters do the same thing. The only difference is that 1x1 (learned) filters ONLY do this across feature-maps where 3x3 filters (say) also consider local spatial correlations.  
  
So, they are used for two reasons:  
* Dimensionality reduction: 
  - When performing larger size convolutions (spatial 3x3 or 5x5...) over a large number of feature maps, bringing down the dimensions in depth (# feature maps) reduces computations dramatically. This is done in GoogLeNet Inception modules (2).  
* Since ReLU will be applied again, it is yet another non-linearity that can be helpful.  
  
  
Put together the parts and from the feature learning part.   
![Featurelearning](/images/featurelearning.png)  

****
## Classification
The output from the convolutional layers represents high-level features in the data.  While that output could be flattened and connected to the output layer, adding a fully-connected layer is a (usually) cheap way of learning non-linear combinations of these features.  
  
Essentially the convolutional layers are providing a meaningful, low-dimensional, and somewhat invariant feature space, and the fully-connected layer is learning a (possibly non-linear) function in that space.  
  
The whole classification part is a fully connected layer, which starts with **flattening step**, then **fully connected layer**(which is layer full of connections to do classification, also called dense layer) and end with **softmax function**.  
  
ANN classifier needs individual features, just like any other classifier. This means it needs a feature vector.  

#### 1. Flattening Step  
Therefore, you need to convert the output of the convolutional part of the CNN into a 1D feature vector, to be used by the ANN part of it. This operation is called flattening. It gets the output of the convolutional layers, flattens all its structure to create a single long feature vector to be used by the dense layer for the final classification.(Like the long vector in the image below.)  
  
![Flattenedvectors](/images/longvectorfeatures.png)  
A list of features vector which is also a list of weights, depending on the threshold that set early, and classify the object by using FC and softmax function.  
  
#### 2. Fully Connected Layer(FC)  
Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. (When activations of all nodes in one layer goes to each and every node in the next layer. When all the nodes in the Lth layer connect to all the nodes in the (L+1)th layer we call these layers fully connected layers.)  
  
![FClayer](/images/fullyconnectedlayers.jpg)    
Their activations can hence be computed with a matrix multiplication followed by a bias offset.   
Apart from classification, adding a fully-connected layer is also a (usually) cheap way of learning non-linear combinations of these features. Most of the features from convolutional and pooling layers may be good for the classification task, but combinations of those features might be even better.  
  
**In this layer, where the weight and bias are like in the normal neural network, use cost to compute the loss function, gradient descent to optimize parameters and reduce cost function.**  

#### 3. Softmax function
The sum of output probabilities from the Fully Connected Layer is 1.  
This is ensured by using the Softmax as the activation function in the output layer of the Fully Connected Layer.  
The [Softmax](http://cs231n.github.io/linear-classify/#softmax) function takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one.  

**Putting the whole Convolutional Neural Networks together:**  
![CNN](/images/cnn.png)  

****
  


 

