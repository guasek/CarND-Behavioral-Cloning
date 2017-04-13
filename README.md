The goal of this project, was to use deep neural networks and convolutional neural networks to clone driving behavior.

I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

I had to record my model controlling a simulation with mobile phone and upload it to youtube https://www.youtube.com/watch?v=fHYvu40IMTc
The reason was every time I tried to record the screen using a tool running on my laptop, the load it induced made the python 
script using the model too slow to successfully steer the car.

In order to solve the problem stated in behavioral cloning project I've started to look for the reference that already dealt with similar
task. One of the project's intro lessons mentioned that Nvidia had already created an NN architecture which was able to 
do the job. That led me to use Nvidia model described in the following whitepaper: 
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf.

After reading entire document I have decided to copy their architecture 1:1. It's characteristics are described below.

The model takes 66x200x3 images as an input and comprises of:

* Convolutional layer with 24 filters of size 5x5 with a stride of 2 and valid padding
* Convolutional layer with 36 filters of size 5x5 with a stride of 2 and valid padding
* Convolutional layer with 48 filters of size 5x5 with a stride of 2 and valid padding
* Convolutional layer with 64 filters of size 3x3 with a stride of 1 and valid padding
* Convolutional layer with 64 filters of size 3x3 with a stride of 1 and valid padding

After each convolutional layer relu activation function is applied.
Then comes a flattening layer followed by 4 fully connected layers. Relu activation function is used after each one.

In order to train the network I've used data provided by udacity. The reason was, data I tried to generated using simulator
lacked consistency. Every time i started recording, steering angle format changed. I just couldn't simply load and use it.

As there weren't sufficient data in udacity dataset to train the model I've generated additional data basing on original
images. The techniques applied were:

* brightness changes - I've changed image colorspace to hsv, adjusted brightness according to a randomly selected value and
converted it back to bgr,
* image shifts - I've moved each image horizontally and added a 0.008 for each pixel to steering angle when I've shifted to the left and
subtracted 0.008 when the shift was to the right,
* I've also used images from side cameras. I've changed the steering angle by 0.25 for both of them,
* I've horizontally flipped images from center camera and used negative value of steering angle

In a preprocessing method I've deleted bottom 30 and top 50 pixels. Thanks to that got rid of a hood and a sky. The reason was
I believe those pixels provided no useful information on how to steer a car.
Then I divided each pixel by 255 so each pixel value was between 0 and 1. After that I've resized each image to
200x66x3 as that was the format nvidia model expected.

I've omitted train test split in training process as I've found selecting model basing on validation loss of mse inefficient.
Models with much worse loss value behaved much better on track than theoretically good ones. Even though my mentor
mentioned there was another, proper implementation of nvidia model with dropout layers after each fully connected layer
I've found them lowering performance of my model.

All the hyperparameters were found trial and error way as I think grid or randomized search would be inefficient due to 
no corelation between the value of a loss function and actual model performance on track.
Out of three methods of overfit reduction - dropout, regularization, additional data generation - I've used only additional 
data generation. 

I've tried to introduce dropout layers in three combinations:
* after 1st fully connected layer
* after 1st and second fully connected layer
* after 1st, 2nd and 3rd fully connected layer

The first option performed the best. Nevertheless the model without dropout behaved much better than any created with them (on both tracks!).

In typical scenario using any form of regularization is necessary in order for a model to avoid simply memorizing training data. In the case
of my model running a car in simulator I've also empirically checked the level of overfit. My solution can successfully navigate
almost entire second track. An exception was the sharpest right turn in a middle of a run.
s
