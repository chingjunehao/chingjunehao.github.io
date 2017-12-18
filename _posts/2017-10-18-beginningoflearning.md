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
(Picture shows the basic CNN model that proposed by Yann LeCun in 1998, after that, all the CNN is built based on it.)
3 good links that can let beginner to understand CNN easily:

* [Youtube video to have a clear picture of CNN by Brandon Rohrer](https://www.youtube.com/watch?v=FmpDIaiMIeA&t=870s)
* [Intro to CNN by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
* [Deeper understanding on CNN by Ujjwal Karn](https://www.kdnuggets.com/2016/11/intuitive-explanation-convolutional-neural-networks.html)

More intense learning on CNN(with code explained):
* [CS231n by Stanford University](http://cs231n.github.io/convolutional-networks/#conv)
Playing around with the filter to get the concept of CNN:
* [](http://setosa.io/ev/image-kernels/)

> ConvNets are good at finding patterns, which is to capture local "spatial"(things that next to one another) patterns of data, so it's > > best to apply on image because the patterns(Pattern in the images mean that the position of the data matters to us)in the image are the > most common one, and if the data fails to be made to look like an image, CNN will be less useful. 
> Depends on the problem and decide to which architecture of the network to be used on the problems.
> CNN is separated into 2 parts:
>> Feature learning
>> Classification
 
