from doctest import OutputChecker
from subprocess import CREATE_NEW_CONSOLE
from turtle import numinput
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import struct
import time as time
import tkinter as tk
from PIL import Image
import cv2
import os

from numpy.random import rand

np.set_printoptions(threshold=np.inf)

# file paths to the MNIST data, in csv format

train_file_path = r"C:\MNIST\train.csv"
test_file_path = r"C:\MNIST\test.csv"

# opening MNIST training & test file
    
train_file = open(train_file_path, "r")
train_list = train_file.readlines()
train_file.close()

test_file = open(test_file_path, "r")
test_list = test_file.readlines()
test_file.close()

# create NN class
"""
The neural net has 2 hidden layers:

1st - 128 neurons
2nd - 64 neurons

the hidden layers use a simgoid activation function
the output layer uses a softmax activation function
"""
class NN:
    def __init__(self, sizes=[784, 128, 64, 10], epochs=3, lr=0.1):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        
        input_layer = sizes[0]
        
        hidden_1 = sizes[1]
        hidden_2 = sizes[2]
        output_layer = sizes[3]
        
        # np.sqrt(1./hidden_n) is used to scale the randomly initiliased weights
        
        self.params = {
            'W1' : (np.random.randn(hidden_1, input_layer) * np.sqrt(1./hidden_1)), # 128 x 784
            'W2' : np.random.randn(hidden_2,hidden_1) * np.sqrt(1./hidden_2), # 64 x 128
            'W3' : np.random.randn(output_layer,hidden_2) * np.sqrt(1./output_layer) # 10 x 64
            }
    
    def feed_forward(self, x_train):
        params = self.params
        
        params['A0'] = x_train.astype(float) # 784 x 1 is the bitmap image
        
        # input_layer to hidden_1
        params['Z1'] = np.dot(params['W1'], params['A0']) # 128 x 1
        params['A1'] = self.sigmoid(params['Z1'])
        
        # hidden_1 to hidden_2
        params['Z2'] = np.dot(params['W2'], params['A1']) # 64 x 1
        params['A2'] = self.sigmoid(params['Z2']) 
        
        # hidden_2 to output_layer
        params['Z3'] = np.dot(params['W3'], params['A2']) # 10 x 1
        params['A3'] = self.softmax(params['Z3']) # softmax
            
        # return output
        return params['A3']
    
    def backpropagate(self, y_train, output):
        params = self.params
        
        change_w = {}
        
        # get error in output layer
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], True) # softmax derivative
        # calculate W3 update matrix
        change_w['W3'] = np.outer(error, params['A2'])
        
        # calculate error in 2nd layer (64 x 1)
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], True)
        # calculate W2 update matrix
        change_w['W2'] = np.outer(error, params['A1'])
        
        # calculate error in 1st layer (128 x 1)
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], True)
        # calculate W1 update matrix
        change_w['W1'] = np.outer(error, params['A0'])
        
        return change_w
    
    def update(self, change_w):
        for key, val in change_w.items():
            self.params[key] -= self.lr * val
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / (( 1 + np.exp(-x) )**2)
        return 1 / ( 1 + np.exp(-x) ) # 1 / (1 + e^-x)
    
    def softmax(self, x, derivative=False):
        exp_x = np.exp(x - np.max(x)) # used for stability
        if derivative:
            return exp_x / np.sum(exp_x, axis = 0) * (1 - exp_x / np.sum(exp_x, axis = 0))
        return exp_x / np.sum(exp_x)
    
    def compute_accuracy(self, test_data):
        predictions = []
        for x in test_data:
            values = x.split(',')
            inputs = np.asarray(values[1:]).astype(float) # exclude first value (the label)
            inputs = (inputs/ 255.0 * 0.99) + 0.01 # avoids overflow; converting the pixel values of 0 - 255 to 0 - 1.0
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99 # setting the labels nueron to 0.99
            output = self.feed_forward(inputs)
            # prediction
            pred = np.argmax(output) # gets the index of the max value
            predictions.append(pred==np.argmax(targets)) # append 1 if the prediction matches the correct label, and 0 if not
            
        return np.mean(predictions)    
    
    def train(self, train_list, test_list):
        for i in range(self.epochs):
            for x in train_list: # iterate over 10,000 training images
                values = x.split(',')
                inputs = np.asarray(values[1:]).astype(float)
                inputs = (inputs/ 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99 # setting the labels neuron to 0.99
                # start learning
                output = self.feed_forward(inputs)
                change_w = self.backpropagate(targets, output)
                self.update(change_w)
            # output accuracy at the end of each epoch
            accuracy = self.compute_accuracy(test_list)
            print(f"accuracy for epoch {i + 1} : {round(accuracy * 100, 2)}%")            
        
    
nn = NN(sizes=[784, 128, 64, 10], epochs=3, lr=0.1)

# output untrained network accuracy

accuracy = nn.compute_accuracy(test_list)
print(f"accuracy of untrained network : {round(accuracy * 100, 2)}%")
print("")

# start training

nn.train(train_list, test_list)
print("")

# -- CREATE CANVAS -- #

root = tk.Tk()
root.title("Draw your digit!")


scale_factor = 8 # determines the scaled up resolution of the canvas
canvas_width = 28 * scale_factor
canvas_height = 28 * scale_factor

canvas = tk.Canvas(root, width=canvas_width, height=canvas_width, bg="white")
canvas.pack()
    

"""
After consulting stack overflow for a while, it seems there is no way to get a
rasterised image of the canvas contents directly in tkinter

the work around proposed was to save the .eps file as a png, and then from there
read the data

not an *amazingly* efficient workaround, however it does the job!
"""
# convert canvas to a rasterized 28 x 28 bitmap imge
def canvas_to_array():
    # save .eps file
    canvas.postscript(file="tmp_canvas.eps",                      
                      colormode="gray",
                      width=canvas_width,
                      height=canvas_height,
                      pagewidth=canvas_width-1,
                      pageheight=canvas_height-1)
    
    # save .png file
    img = Image.open("tmp_canvas.eps")
    img.save("tmp_canvas.png", "png")
    
    # open .png file
    data = cv2.imread("tmp_canvas.png")
    
    data -= 255 # invert image
    
    # the data is stored in RGB values, so I take every third value (it is arbitrary, I just decided on the blue value)
    flat_data = data.flatten()
    flat_data = flat_data[0::3]
   
    canvas_data = np.array(flat_data, dtype=int)
    canvas_data = np.reshape(canvas_data, (canvas_width,canvas_width))
    
    formatted_data = np.zeros((28,28), dtype=float)
    
    # downsample to 28 x 28
    
    summed_pixels = 0
    average_pixel = 0
    
    # split canvas up into (scale_factor) x (scale_factor) chunks
    
    for i in range(0, canvas_width, scale_factor):
        for j in range(0, canvas_width, scale_factor):
            
            # average the pixel data over that chunk
            
            for p in range(scale_factor):
                for q in range(scale_factor):
                    summed_pixels += canvas_data[i+p,j+q] # sum every pixel value
            # average value
            average_pixel = summed_pixels / (scale_factor*scale_factor)
            # append
            formatted_data[int(i / scale_factor),int(j / scale_factor)] = average_pixel
            summed_pixels = 0
    
    formatted_data = formatted_data * 0.99 + 0.1 # set range to 0 - 1.0


    output = nn.feed_forward(formatted_data.reshape(784,1)) # feed in the drawn image
    
    print(f"the network classifies your drawn image as the number {np.argmax(output)}")
    print(f"the network classifies this with {int(round(np.max(output),2) * 100)}% certainty")
    print("")
    for i in range(output.size):
        x = output[i][0]
        print(f"{i} certainty : {int(round(x,2) * 100)}%")
    
    plt.imshow(formatted_data, cmap='gray')
    plt.title("28x28 downsampled digit")
    plt.show()

# clear canvas
def clear_canvas():
    canvas.delete("all")
   
def draw(event):
    # get the mouse's current position
    x, y = event.x, event.y
    r = scale_factor  # brush radius
    
    # create a small circle at the mouse position
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
    
# capture button
capture_button = tk.Button(root, text="capture!", command=canvas_to_array)
capture_button.pack()

# clear button
clear_button = tk.Button(root, text="clear", command=clear_canvas)
clear_button.pack()

# bind mouse movement to drawing
canvas.bind("<B1-Motion>", draw)

root.mainloop()