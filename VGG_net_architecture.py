#Machine learning project Part 1 
#Age estimation using convolutional neural networks based on vgg net architecture
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
    dir_name='C:/Users/jchin/Desktop/ml project/wiki.tar/wiki'
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
print("reached here")            
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
######## Convolutional neural networks model based on VGG net architecture ##########


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

def convolutional_network(x, weights, biases, dropout):
    
    # Reshape face image : input size-256*256*3=196608, after reshape-3D[256,256,3]  
    x = tf.reshape(x, shape=[-1, 256, 256, 3])
    
    
    # Design of Convolution Layer 1
    # Input x shape : [256x256x3], filter : [5x5x3,32] , bias : [32] , convolutional_layer1 : [256x256x32]
    # Relu1 : [256x256x32]
    #maxpool1:[128x128x32]
    #lrn1:[128x128x32]
    
    convolutional_layer1 = filter_conv(x, conv1_weights, conv1_bias)
    relu1 = tf.nn.relu(convolutional_layer1)
    maxpool1 = maxpool_layer(relu1)
    lrn1 = tf.nn.local_response_normalization(maxpool1, alpha=0.0001, beta=0.75)
    
    # Design of Convolution Layer 2
    # Input shape : [128x128x32], filter : [3x3x64,64] , bias : [64] , convolutional_layer2 : [128x128x64]
    # Relu2 : [128x128x64]
    #maxpool2:[64x64x64]
    #lrn2:[64x64x64]
    
        
    convolutional_layer2 = filter_conv(lrn1, conv2_weights, conv2_bias)
    relu2 = tf.nn.relu(convolutional_layer2)
    maxpool2 = maxpool_layer(relu2)
    lrn2 = tf.nn.local_response_normalization(maxpool2, alpha=0.0001, beta=0.75)
    
    
    # Design of Convolution Layer 3
    # Input shape : [64x64x64], filter : [3x3x64,128] , bias : [128] , convolutional_layer3 : [64x64x128]
    # Relu3 : [64x64x128]
    #maxpool3:[32x32x128]
    #lrn3:[32x32x128]
    
        
    convolutional_layer3 = filter_conv(lrn2, conv3_weights, conv3_bias)
    relu3 = tf.nn.relu(convolutional_layer3)
    maxpool3 = maxpool_layer(relu3)
    lrn3 = tf.nn.local_response_normalization(maxpool3, alpha=0.0001, beta=0.75)
    

    # Fully connected layer 1 (FC1)
    #input :lrn3 of shape 32x32x128 will be reshaped into flat array of length [32*32*128,1]=[131072,1] 
    #FC1 final shape:[1024] 
    fc1_input = tf.reshape(lrn3, [-1, fc1_weights.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1_input, fc1_weights), fc1_bias)
    fc1_relu = tf.nn.relu(fc1)
    #Dropout ratio =.75
    fc1_dropout = tf.nn.dropout(fc1_relu, dropout)

    # Output, Fully connected layer 2 (FC2): 1024 neurons will be connected to output 100 labels in fully connected layer 2
    out = tf.add(tf.matmul(fc1_dropout, fc2_weights), fc2_bias)
    return out


#weights

# Convolutional layer 1 weights : 5x5x3, 32 filters
conv1_weights = tf.Variable(tf.random_normal([5,5,3,32]),name="conv1_weights") 

# Convolutional layer 2 weights : 3x3x3, 64 filters
conv2_weights = tf.Variable(tf.random_normal([3,3,32,64]),name="conv2_weights")

# Convolutional layer 3 weights : 3x3x3, 128 filters
conv3_weights = tf.Variable(tf.random_normal([3,3,64,128]),name="conv3_weights")

# Fully connected layer 1 weights : 32*32*128, 1024 neurons
fc1_weights = tf.Variable(tf.random_normal([32*32*128,1024]),name="fc1_weights") 

# Fully connected layer 2 weights : 1024, output= 100 classes
fc2_weights = tf.Variable(tf.random_normal([1024,n_classes]),name="fc2_weights") 


#bias

conv1_bias= tf.Variable(tf.random_normal([32]))
conv2_bias= tf.Variable(tf.random_normal([64]))
conv3_bias= tf.Variable(tf.random_normal([128]))
fc1_bias= tf.Variable(tf.random_normal([1024]))
fc2_bias= tf.Variable(tf.random_normal([n_classes]))


# Feed input to the CNN model 
predicted_age = convolutional_network(x, conv1_weights, conv1_bias, keep_prob)

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

print("end")
