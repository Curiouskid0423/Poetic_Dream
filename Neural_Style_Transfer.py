import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import imageio


#load the model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

#FUNCTION: Computer Content Cost
def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(tf.transpose(a_C, [0,3,1,2]), [m, n_C, -1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, [0,3,1,2]), [m, n_C, -1])

    J_content = tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled)) / (4*n_H*n_W*n_C)

    return J_content

#-----------------------------------------------------------------------------------------------
# FUNCTION: gram_matrix
def gram_matrix(A):

    GA = tf.matmul(A, A, transpose_b = True)

    return GA

#-----------------------------------------------------------------------------------------------
#  FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value - style cost
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S, [0,3,1,2]), [m, n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G, [0,3,1,2]), [m, n_C, n_H*n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.squared_difference(GS, GG)) / (4 * n_C**2 * (n_H*n_W)**2)

    return J_style_layer

#-----------------------------------------------------------------------------------------------
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]

        a_S = sess.run(out)
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

#-----------------------------------------------------------------------------------------------
# FUNCTION: total_cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style

    return J
#-----------------------------------------------------------------------------------------------
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

content_image = imageio.imread("images/4x3_man_alone.jpg")
content_image = reshape_and_normalize_image(content_image)
style_image = imageio.imread("images/4x3_rainPrincess.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)
# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))
# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, alpha = 10, beta = 40)
# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 400):

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%10 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image
#-----------------------------------------------------------------------------------------------
model_nn(sess, generated_image)
