#Machine learning project Part 2
#Age estimation using convolutional neural networks based on GoogLeNet architecture
#data set - IMDB and WIKI dataset (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
#Training data = 15000 images, testing data= 1000 images



#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import *
from PIL import Image
import os


########     Data preprocessing    ########


#Extracting RGB pixels into 3d array from the images
#Images are resized to 256x256 and then extract RGB values 
#Applying mean subtract to preprocess the data.


#Picpath.txt consists of the path to the face images which is extracted from matlab code provided(sample: 17/10000217_1981-05-05_2009.jpg)
#The extracted data(picpaths.txt) from the matlab has been included in the zip file 
#Expected to give path 

x1=np.genfromtxt('C:/data/project/picpaths.txt',dtype='str')
x_input=x1[:16000]



def convert_faceimage_to_RGB_array(image_path):

  with Image.open(image_path) as image: 
    
    #pixel extraction and resizing into 256x256x3 arrays
    image = image.resize((256,256),Image.ANTIALIAS)  
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    
    
    #Mean subtraction is used for data preprocessing and to decrease compuatational overhead
 
    if (len(im_arr) == 256*256*3):
        meanimage=np.mean(im_arr,axis=0)
        meanimage=meanimage.astype(int)
        im_arr -= meanimage
            
    else:
        im_arr=np.zeros((256*256*3,1))
      
  return im_arr


def append_quotes(s1):
    return "'" + s1 + "'" 


list_of_3D_RGB_pixel_images=[]
count=0


#1.Create list of consisting of image paths by combining the directory address(where images are present) and image name(xyz.jpeg) 
#2.Create list of 3d RGB pixels by passing each image path to the function : 'convert_faceimage_to_RGB_array'


for i in range(0,len(x_input)):
    z = np.copy(x_input[i])
    z=np.array2string(z)
    z=z.strip("'")
    #expected to give images path here
    dir_name='C:/data/project/wiki'
    base_filename=z
    temppath= os.path.join(dir_name, base_filename)

    if i==0:
        l0=convert_faceimage_to_RGB_array(temppath)
        list_of_3D_RGB_pixel_images=np.reshape(l0, (1,len(l0) ))

    else:
        l1=convert_faceimage_to_RGB_array(temppath)
        l1=np.reshape(l1, (1,len(l1) ))
        list_of_3D_RGB_pixel_images=np.vstack((list_of_3D_RGB_pixel_images,l1))
                    



#Extracting ages for the respective face images
y1=np.genfromtxt('C:/data/project/picage.txt',delimiter=',')
y_input=y1[:16000]
#k=jpg_image_to_array('C:/data/project/abcd.jpg')
agelist = np.zeros(shape=[len(y_input),100])

#generating age labels[example: If age=3, then it is converted to [0 0 1 0 0 .... (100 labels)]
for i in range(0,len(y_input)):
    temp1=y_input[i]
    temp1=temp1.astype(int)
    if temp1>100:
        temp1=100
    agelist[i,temp1-1]=1
 

            
#  
######## Convolutional neural networks model based on GoogLeNet net architecture ##########


# Network Parameters

#RGB : 3 channels [256,256,3], n_classes: age labels
# Dropout ratio
image_size = 256*256*3 
age_labels = 100 
dropout = 0.75  
n_classes=100


#Optimization Parameters
no_of_iterations = 500
input_feed_size = 30
optimization_rate = 0.001
check_accuracy_stepsize=15

#Tensorflow placeholder for inputs x: [256x256x3], y : [100]
x = tf.placeholder(tf.float32, [None, image_size])
y = tf.placeholder(tf.float32, [None, age_labels])
keep_prob = tf.placeholder(tf.float32) 


# Function for convolution of input and filter, adding bias term(w.X + b)
def filter_conv(input_x, Weights, bias):
    stride_length=1
    #convolution w*x
    x = tf.nn.conv2d(input_x, Weights, strides=[1, stride_length, stride_length, 1], padding='SAME')
    #add bias term (w*x + b)
    x = tf.nn.bias_add(x, bias)
    return x

#maxpool 
def maxpool_layer(x):
    stride_length=1
    return tf.nn.max_pool(x, ksize=[1, stride_length, stride_length, 1], 
                          strides=[1, stride_length, stride_length, 1],padding='SAME')


# convolution neural network model
#3 conv layers and 2 fully connected(fc) layers are used 
#Each convolution layer is passed to relu layer and max pool layer followed by normalization
#The resultant has been passes to next convolution layer 
#Fc layer 1 has 1024 neurons and Fc layer 2 which is the final layer contains 100 labels.

def GoogLeNet_architecture(x, weights, biases, dropout):
    
    # Reshape face image : input size-256*256*3=196608, after reshape-3D[256,256,3]  
    x = tf.reshape(x, shape=[-1, 256, 256, 3])
    
    
    # Design of Layer 1 : 1x1 conv
    # Input x shape : [256x256x3], filter : [1x1x3,32] , bias : [32] , layer1_conv_1x1 : [256x256x32]
    # layer1_relu1 : [256x256x32]
    #layer1_lrn1:[256x256x32]
    
    layer1_conv_1x1 = filter_conv(x, layer1_conv1x1_weights, layer1_conv1x1_bias)
    layer1_relu1x1 = tf.nn.relu(layer1_conv_1x1)
    layer1_lrn1x1 = tf.nn.local_response_normalization(layer1_relu1x1, alpha=0.0001, beta=0.75)
    
    # Design of Layer 1 : 3x3 max pooling
    # Input shape : [256x256x3], output of maxpool layer : [128x128x3]
        
    layer1_maxpool = maxpool_layer(x)
    layer1_relu3x3= tf.nn.relu(layer1_maxpool)
    
        
    # Design of  Layer 2: 1x1 conv
    # Input shape : [128x128x3], filter : [1x1x3,64] , bias : [64] , convolutional_layer output: [128x128x64]
    # layer2_relu1x1 : [128x128x64]
    #layer2_lrn1x1:[128x128x64]
    
        
    layer2_conv_1x1 = filter_conv(layer1_relu3x3, layer2_conv1x1_weights, layer2_conv1x1_bias)
    layer2_relu1x1 = tf.nn.relu(layer2_conv_1x1)
    layer2_lrn1x1 = tf.nn.local_response_normalization(layer2_relu1x1, alpha=0.0001, beta=0.75)
    
    
    # Design of  Layer 2: 3x3 conv
    # Input shape : [256x256x32], filter : [3x3x32,64] , bias : [64] , convolutional_layer output: [256x256x64]
    # layer2_relu3x3 : [256x256x64]
    #layer2_maxpool3x3 : [128x128x64]
    #layer2_lrn3x3:[128x128x64]
    
        
    layer2_conv_3x3 = filter_conv(layer1_lrn1x1, layer2_conv3x3_weights, layer2_conv3x3_bias)
    layer2_relu3x3 = tf.nn.relu(layer2_conv_3x3)
    layer2_maxpool3x3 = maxpool_layer(layer2_relu3x3)
    layer2_lrn3x3 = tf.nn.local_response_normalization(layer2_maxpool3x3, alpha=0.0001, beta=0.75)
    
    
        
    # Design of  Layer 2: 5x5 conv
    # Input shape : [256x256x32], filter : [5x5x32,64] , bias : [64] , convolutional_layer output : [256x256x64]
    # layer2_relu5x5 : [256x256x64]
    #layer2_maxpool3x3 : [128x128x64]
    #layer2_lrn5x5:[128x128x64]
    
        
    layer2_conv_5x5 = filter_conv(layer1_lrn1x1, layer2_conv5x5_weights, layer2_conv5x5_bias)
    layer2_relu5x5 = tf.nn.relu(layer2_conv_5x5)
    layer2_maxpool5x5 = maxpool_layer(layer2_relu5x5)
    layer2_lrn5x5 = tf.nn.local_response_normalization(layer2_maxpool5x5, alpha=0.0001, beta=0.75)
    
    #Inception layer
    #Filter concatenation of convolution layers 1x1,3x3, 5x5 
    #Relu and Dropout are applied to inception layer 
    
    inception = tf.nn.relu(tf.concat([layer1_lrn1x1,layer2_lrn1x1 ,layer2_lrn3x3, layer2_lrn5x5],3))
    inception_relu = tf.nn.relu(inception)
    #Dropout ratio =.75
    out = tf.nn.dropout(inception_relu, dropout)
    
    return out


#weights

# Layer 1:  1x1 conv weights : 1x1x3, 32 filters
layer1_conv1x1_weights = tf.Variable(0.01*tf.random_normal([1,1,3,32]),name="layer1_conv1x1_weights") 

# Layer 2:   1x1 conv weights : 1x1x3, 64 filters
layer2_conv1x1_weights = tf.Variable(0.01*tf.random_normal([1,1,3,64]),name="layer2_conv1x1_weights")

# Layer 2:   3x3 conv weights : 3x3x32, 64 filters
layer2_conv3x3_weights = tf.Variable(0.01*tf.random_normal([3,3,32,64]),name="layer2_conv3x3_weights")

# Layer 2:   5x5 conv weights : 5x5x32, 64 filters
layer2_conv5x5_weights = tf.Variable(0.01*tf.random_normal([5,5,32,64]),name="layer2_conv5x5_weights")



#bias

layer1_conv1x1_bias= tf.Variable(0.01*tf.random_normal([32]))
layer2_conv1x1_bias= tf.Variable(0.01*tf.random_normal([64]))
layer2_conv3x3_bias= tf.Variable(0.01*tf.random_normal([64]))
layer2_conv5x5_bias= tf.Variable(0.01*tf.random_normal([64]))



# Feed input to the CNN model 
predicted_age = GoogLeNet_architecture(x, layer1_conv1x1_weights, layer1_conv1x1_bias, keep_prob)

# Optimization 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_age, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=optimization_rate).minimize(cost)

# Evaluation
correct_pred = tf.equal(tf.argmax(predicted_age, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Tensorflow variable initialization
init = tf.global_variables_initializer()

# Graph initialization
with tf.Session() as sess:
    sess.run(init)
    step_size = 1   
    
    # Optimization using Adam optimizer(with gradient descent method)
    while step_size * input_feed_size < no_of_iterations:
        batch_x=list_of_3D_RGB_pixel_images[(input_feed_size*(step_size-1)):(input_feed_size*(step_size))]
        batch_y=agelist[(input_feed_size*(step_size-1)):(input_feed_size*(step_size))]
        
        # Run optimization op (backprop)
        a1,b1=sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})  
        if step % check_accuracy_stepsize == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y,keep_prob: 1.})
            print (" cost and accuracy at iteration %s is %s, %s" %(step,loss,acc))  
        step_size += 1
 
    #accuracy for 1000 images
    print ("Accuracy:", sess.run(accuracy, feed_dict={x: list_of_3D_RGB_pixel_images[15000:16000],y: agelist[15000:16000],keep_prob: 1.}))

