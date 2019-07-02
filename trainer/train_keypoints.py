import tensorflow as tf 
import os
import sys
import gzip
import cv2
import json
import numpy as np
import time

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = 80
IMAGE_SIZE_W = 578
IMAGE_SIZE_H = 310
NUM_CHANNELS = 3
SEED = 48923

TRAIN_SIZE = 3
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

NUM_KEYPOINTS = 7

EVAL_FREQUENCY = 1

NUM_EPOCHS = 10

N_CHANNELS_1 = 64
N_CHANNELS_2 = 128
N_CHANNELS_3 = 256
N_CHANNELS_4 = 512

conv1_weights = tf.Variable(tf.truncated_normal([7,7,NUM_CHANNELS,64],stddev=0.1,seed=SEED,dtype=tf.float32))
conv1_biases = tf.Variable(tf.zeros([64]))

resnet1_conv_weights = tf.Variable(tf.truncated_normal([3,3,64,64],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet1_conv_biases = tf.Variable(tf.zeros([64]))

resnet2_conv_weights = tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet2_conv_biases = tf.Variable(tf.zeros([128]))
resnet2_conv2_weights = tf.Variable(tf.truncated_normal([3,3,128,128],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet2_conv2_biases = tf.Variable(tf.zeros([128]))

resnet3_conv_weights = tf.Variable(tf.truncated_normal([3,3,128,256],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet3_conv_biases = tf.Variable(tf.zeros([256]))
resnet3_conv2_weights = tf.Variable(tf.truncated_normal([3,3,256,256],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet3_conv2_biases = tf.Variable(tf.zeros([256]))

resnet4_conv_weights = tf.Variable(tf.truncated_normal([3,3,256,512],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet4_conv_biases = tf.Variable(tf.zeros([512]))
resnet4_conv2_weights = tf.Variable(tf.truncated_normal([3,3,512,512],stddev=0.1,seed=SEED,dtype=tf.float32))
resnet4_conv2_biases = tf.Variable(tf.zeros([512]))

fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE*N_CHANNELS_4,512],stddev=0.1,seed=SEED,dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1,shape=[512]))

fc2_weights = tf.Variable(tf.truncated_normal([512,14],stddev=0.1,seed=SEED,dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(0.1,shape=[14]))

base_path = "/home/mujahid/Tensorflow/workspace/landmark_training/images"
data_dir = "/data"
label_dir = "/labels"
def prep_data():
    data = []
    data_path = base_path + "/" + data_dir
    for file in os.listdir(data_path):
        data.append(cv2.imread(data_path+"/"+file))
    return np.array(data)
    
def prep_points_data():
    labels = []
    label_path = base_path + label_dir + "/labels.json"
    with open(label_path, 'r') as myfile:
        data=myfile.read()
        obj = json.loads(data)
        for ob in obj:
            #print ob
            #print obj[ob]['filename']
            arr = []
            for reg in sorted(obj[ob]['regions']):
                reg = obj[ob]['regions'][reg]['shape_attributes']
                cx = reg['cx']
                cy = reg['cy']
                arr.append(cx)
                arr.append(cy)
            #print arr
            labels.append(arr)
            #print("===================================================")
    return np.array(labels)
    

def eucl_dist(point1,point2):
    return tf.sqrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )

def cross_ratio_2d(point1,point2,point3,point4):
    return (eucl_dist(point1,point3) * eucl_dist(point2,point4)) / (eucl_dist(point1,point4) * eucl_dist(point2,point3))


def model(X,train=False):
    conv1 = tf.nn.conv2d(X,conv1_weights,strides=[1,1,1,1],padding="SAME")
    bn1 = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(tf.nn.bias_add(bn1,conv1_biases))
    
    res1_conv1 = tf.nn.conv2d(relu1,resnet1_conv_weights,strides=[1,1,1,1],padding="SAME")
    res1_bn1 = tf.layers.batch_normalization(res1_conv1)
    res1_relu1 = tf.nn.relu(tf.nn.bias_add(res1_bn1,resnet1_conv_biases))
    res1_conv2 = tf.nn.conv2d(res1_relu1,resnet1_conv_weights,strides=[1,1,1,1],padding="SAME")
    res1_bn2 = tf.layers.batch_normalization(res1_conv2)
    res1 = tf.nn.relu(tf.nn.bias_add(res1_bn2,resnet1_conv_biases) + res1_bn1)
    
    res2_conv1 = tf.nn.conv2d(res1,resnet2_conv_weights,strides=[1,1,1,1],padding="SAME")
    res2_bn1 = tf.layers.batch_normalization(res2_conv1)
    res2_relu1 = tf.nn.relu(tf.nn.bias_add(res2_bn1,resnet2_conv_biases))
    res2_conv2 = tf.nn.conv2d(res2_relu1,resnet2_conv2_weights,strides=[1,1,1,1],padding="SAME")
    res2_bn2 = tf.layers.batch_normalization(res2_conv2)
    res2 = tf.nn.relu(tf.nn.bias_add(res2_bn2,resnet2_conv2_biases) + res2_bn1)
    
    res3_conv1 = tf.nn.conv2d(res2,resnet3_conv_weights,strides=[1,1,1,1],padding="SAME")
    res3_bn1 = tf.layers.batch_normalization(res3_conv1)
    res3_relu1 = tf.nn.relu(tf.nn.bias_add(res3_bn1,resnet3_conv_biases))
    res3_conv2 = tf.nn.conv2d(res3_relu1,resnet3_conv2_weights,strides=[1,1,1,1],padding="SAME")
    res3_bn2 = tf.layers.batch_normalization(res3_conv2)
    res3 = tf.nn.relu(tf.nn.bias_add(res3_bn2,resnet3_conv2_biases) + res3_bn1)
    
    res4_conv1 = tf.nn.conv2d(res3,resnet4_conv_weights,strides=[1,1,1,1],padding="SAME")
    res4_bn1 = tf.layers.batch_normalization(res4_conv1)
    res4_relu1 = tf.nn.relu(tf.nn.bias_add(res4_bn1,resnet4_conv_biases))
    res4_conv2 = tf.nn.conv2d(res4_relu1,resnet4_conv2_weights,strides=[1,1,1,1],padding="SAME")
    res4_bn2 = tf.layers.batch_normalization(res4_conv2)
    res4 = tf.nn.relu(tf.nn.bias_add(res4_bn2,resnet4_conv2_biases) + res4_bn1)
    
    res4_shape = res4.shape
    reshape = tf.reshape(res4,[res4_shape[0], res4_shape[1]*res4_shape[2]*res4_shape[3] ])
    
    hidden = tf.matmul(reshape,fc1_weights) + fc1_biases
    hidden_bn = tf.layers.batch_normalization(hidden)
    hidden_relu = tf.nn.relu(hidden_bn)
    
    outputs =  tf.matmul(hidden_relu,fc2_weights) + fc2_biases
    return outputs
    
prep_data = prep_data()
prep_labels = prep_points_data()

# test_data = prep_data[0:80,...]
# test_labels = prep_labels[0:80,...]
# validation_data = prep_data[80:100, ...]
# validation_labels = prep_labels[80:100, ...]
train_data = prep_data
train_labels = prep_labels
    
train_data_node = tf.placeholder(shape=[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],dtype=tf.float32)
#Points Ground Truth train_labels_node
real = tf.placeholder(shape=[BATCH_SIZE,2*NUM_KEYPOINTS],dtype=tf.float32)
eval_data_node = tf.placeholder(shape=[EVAL_BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],dtype=tf.float32)

cross_ratio_3d = 1.39408 # How to calculate cross_ratio_3d
cr_gamma = 0.0001
num_epochs = NUM_EPOCHS
train_size = TRAIN_SIZE

#Points From Training
logits = model(train_data_node)

loss = 0.0
for item in range(BATCH_SIZE):
    for i in range(NUM_KEYPOINTS):
        loss += (logits[item][i*2] - real[item][(i*2)+1])**2 + (logits[item][i*2] - real[item][(i*2)+1])**2
    + cr_gamma * (cross_ratio_2d([ logits[item][0],logits[item][1] ],
                [ logits[item][2],logits[item][3] ],
                [ logits[item][4],logits[item][5] ],
                [ logits[item][6],logits[item][7] ]) - cross_ratio_3d)**2
    + cr_gamma * (cross_ratio_2d([ logits[item][0], logits[item][1]  ],
                [ logits[item][8],logits[item][9] ],
                [ logits[item][10],logits[item][11] ],
                [ logits[item][12],logits[item][13] ]) - cross_ratio_3d)**2

            
regularizers = tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases)
loss += 5e-4 * regularizers
batch = tf.Variable(0, dtype=tf.float32)
learning_rate = tf.train.exponential_decay(
    0.01,                # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    train_size,          # Decay step.
    0.95,                # Decay rate.
    staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate,0.01).minimize(loss)
    
start_time = time.time()    
with tf.Session() as sess:
    print("global_var_0")
    tf.global_variables_initializer().run()
    print("global_var_1")
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset+BATCH_SIZE), ...] 
        feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
        sess.run(optimizer,feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
            l, lr, predictions = sess.run([loss, learning_rate,train_prediction],feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size,1000 * elapsed_time / EVAL_FREQUENCY))
# print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
# print('Minibatch error: %.1f%%'% error_rate(predictions, batch_labels)