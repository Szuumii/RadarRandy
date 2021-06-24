# RadarRandy
Code for my Bachelor's Degree Thesis

To run this project yot need to modify the python path enviornment variable to include absolute path to root of the project:

```python
export PYTHONPATH=$PYTHONPATH:/home/..../RadarRandy
```

### Abstract

Place recognition remains an open problem to this day due to the number of
issues that are at the core of this concept. This paper presents a learning-basedmethod for
computing a discriminative radar scan image descriptor for place recognition purposes.
Place recognitionmethods based on radar scans are a novelty. Recent advancements in
this area are caused by the appearance of the datasets that enabled new developments.
In the area of visual place recognition, deep learning methods are providing substantial
results, outperforming other methods. In my solution, I plan to use a convolutional neural
network as a backbone of global descriptor generation to study its reliability with regards
to the radar scan images. In the paper you will find the approach to the problem of
place recognition with the usage of radar scan images, description of the data set and any
required preprocessing, creation of the trainingmethod, and final evaluation. Results may
be used as a baseline for further developments in this growing subdomain.

### Keywords
Place Recognition, Convolutional neural network, Global descriptor, Radar
scan images

## Introduction
Applying deep learning methods to solve computer vision problems is an area of active development. There exists wide range of methods for classification, semantic segmen-tation, local and global features extraction. I focus my attention on finding a discrimi-native, low-dimensional image radar descriptors for place recognition purposes. Local-ization is performed by searching through the database with a radar images and their respective well known positions and finding the descriptor closest to our image of the unknown location. This idea of place recognition method is widely used in robotics, autonomous driving and augmented reality.

## Dataset

My solution uses learning-based method to improve it’s accuracy and reliability. [MulRan](https://sites.google.com/view/mulran-pr/home) dataset provides sparse long-range intensity data using radar for localization an mapping of urban areas created with the usage of robust mobile platform. This dataset has been selected to develop my so-lution because it gives me all necessary data that are required for my architecture to learn. This section describes the MulRan dataset and it’s methodology.

### Dataset Equipment

Equipment used to create readings from my dataset are presented on Fig. 1. Different
sensor including radar, LiDAR as well as GPS and other positioning systems have been
mounted on top of the vehicle to increase it’s mobility and robustness. This platform is visible in Figure below.

<center>
<img src="https://github.com/Szuumii/RadarRandy/blob/main/images/mobile-platform.PNG" width="400" height="400">
</center>


### Radar data

In the dataset we can find two types of radar data. One of them is 1D intensity arrays
(raw data) taken from each angle. Data are accumulated by 1 sweep (400 angle bins)
and timestamps are added to reduce communication load. Other type of data are 360°
polar images. Sample of polar image data is presented below.

<center>
<img src="https://github.com/Szuumii/RadarRandy/blob/main/images/polarExample.png" width="400" height="400">
</center>

### Trajectories

Another key attribute with regards to my method is the position of each image that
corresponds to real life location. Due to the complexity of urban areas Global Positioning
System (GPS) data finds very sporadic usage. It’s really difficult to extract baseline
position of the vehicle in this type of environment. Typical GPS and high-precision
virtual reference station (VRS)-GPS are unable to estimate global location due to those
complexities. In the dataset the position were established partially by VRS-GPS data,
fiber optic gyro (FOG) data and graph SLAM were used to estimate baseline of the
position. The reference grid with positions of different data gathering locations is visible
in Figure below.

<center>
<img src="https://github.com/Szuumii/RadarRandy/blob/main/images/position-grid.PNG">
</center>

## Solution Design

The goal is to compute a generalizable global descriptor from the input radar image.
This section describes the proposed architecture and training process of the network
computation.

### Network Architecture
In my methodology I plan on utilizing capabilities of neural networks to create the representations
of radar images. Each representation is a N-length vector ( typically 128
or 256 ) that is created at the last layer of my Neural Network Architecture. I’ve decided
to pre-trained model to establish a baseline and work my way from there. Pre-trained
model is a network that have already been trained on a huge dataset. All I need to do is
to swap the last fully connected layer and fine tune it for our purposes. This is a very
common process in many computer vision projects. The model I’ve picked is ResNet in 34-layered variant.

### Training

To train my network I use a deep metric learning approach [5] with a triplet
margin loss defined as:
<center>
𝐿(𝑎 , 𝑝 , 𝑛) = max⁡{𝑑(𝑎 , 𝑝) − ⁡𝑑(𝑎 , 𝑛) + 𝑚, 0}
</center>

Where 𝑑(𝑥, 𝑦) =⁡∥ 𝑥 − 𝑦 ∥2 is an Euclidean distance between embeddings x and y; a ,
p , n are embeddings (representations) of an anchor a, positive p and a negative n elements
and m is a margin hyperparameter. The loss function is minimized using a stochastic gradient descent approach with Adam optimizer. This method have been also used with regards to person re-identification and has given
very promising results. At the begging of each epoch I partition the dataset into random
batches. A batch of size n is constructed by sampling n/2 pairs of elements that are
known to be positive. I define positive images are those that are withing 10 meters from
each other. They should present similarly structured images. After each batch is constructed,
I compute the two n x n boolean masks, one indicating other positive elements
within the batch and other that represents elements that are known to be negative. Negative
images are known to be further than 50 meters from each other which should give
them dissimilar structure. Then the batch is fed to the network to compute embeddings.
Using our Boolean mask we mine hardest positive and hardest negative examples and
construct triplet that is supplied to our triplet loss function which then applies backpropagation.
Mined hard triplet can be seen in Figure below. This triplet has been extracted
from the MulRan dataset. Triplet loss function “favours” similar embeddings generation
for the images that are know to be positive and “disapproves” similarity in images
that are known to be negative.

<center>
<img src="https://github.com/Szuumii/RadarRandy/blob/main/images/learning-triplet.png">
</center>

### Evaluation Methodology

For the evaluation purposes we need to create two specific datasets. One of them consists
of radar images, their calculated representations and real-life location of each image.
I’ll use it as a database with a reference images. Other test dataset consists of radar
scan images and their corresponding location. We randomly choose the image from the
dataset, then we create it’s representation by forwarding it through the network and
saving it’s embedding. With the embedding generated we search through our database
set and extract images that have smallest distance to our representation. If our return
images have the position that is truly positive ( within 10 meters ) we define our evaluation
as success. We repeat our image selection k times and create the accuracy metric
off of that.

<center>
<img src="https://github.com/Szuumii/RadarRandy/blob/main/images/location-method.PNG">
</center>