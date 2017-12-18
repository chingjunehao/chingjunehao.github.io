---
layout: post
title: Understanding Convolutional Neural Networks
published: true
comments: true
permalink: /learningblog/
category: learning
---

_This post is written by summarizing few blog post that explained CNN with adding some of my ideas._

![LeNet](/images/LeNet.png)  
_Picture shows the basic CNN model that proposed by Yann LeCun in 1998, after that, all the CNN is built based on it._
----
3 good links that can let beginner to understand CNN easily:

* [Youtube video to have a clear picture of CNN by Brandon Rohrer](https://www.youtube.com/watch?v=FmpDIaiMIeA&t=870s)
* [Intro to CNN by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
* [Deeper understanding on CNN by Ujjwal Karn](https://www.kdnuggets.com/2016/11/intuitive-explanation-convolutional-neural-networks.html)

More intense learning on CNN(with code explained):
* [CS231n by Stanford University](http://cs231n.github.io/convolutional-networks/#conv)
Playing around with the filter to get the concept of CNN:
* [Image Kernel by Victor Powell](http://setosa.io/ev/image-kernels/)

> ConvNets are good at finding patterns, which is to capture local "spatial"(things that next to one another) patterns of data, so it's  best to apply on image because the patterns(Pattern in the images mean that the position of the data matters to us)in the image are the  most common one, and if the data fails to be made to look like an image, CNN will be less useful.  
**Depends on the problem and decide to which architecture of the network to be used on the problems.**

## Convolutional Neural Networks is separated into 2 parts:
* **Feature learning**
* **Classification**

> _Images are a matrix of pixel values._  
[**Channel**](https://en.wikipedia.org/wiki/Channel_(digital_image)) is a conventional term used to refer to a certain component of an image. An image from a standard digital camera will have three channels – red, green and blue(RGB) – you can imagine those as three 2d-matrices stacked over each other (one for each color), each having pixel values in the range 0 to 255.  

> While a grayscale image, has just one channel. The value of each pixel in the matrix will range from 0 to 255 – zero indicating black and 255 indicating white.  

### Feature Learning
#### 1. Convolution Part
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

>> _Time takes to run the filter = ((height*width)of image)/stride_  

Good example to illustrate the working of filter on image.  
![Filter on image](/images/filteronimage.png)  
As you can see, there are few types of filter, which to scan through the image and convolve to the next layer, which with depth(equal to the number of filters used).  




