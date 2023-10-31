import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import cv2
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image
import argparse
from cnn.neural_network import CNN
from keras.utils import np_utils
from keras.optimizers import SGD
# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml


dim = 28

# Defing and compile the SGD optimizer and CNN model
print('\n Compiling model...')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
                Saved_Weights_Path='model/cnn_weights_4.hdf5')
clf.compile(loss="categorical_crossentropy",
            optimizer=sgd, metrics=["accuracy"])


sudoku_a = cv2.imread(
    'datasets\sudoku_dataset-master\original\image1088.original.jpg')


# Preprocessing image to be read
sudoku_a = cv2.resize(sudoku_a, (1000, 1000))
plt.imshow(sudoku_a)

# function to greyscale, blur and change the receptive threshold of image


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    #blur = cv2.bilateralFilter(gray,9,75,75)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold_img


threshold = preprocess(sudoku_a)


# Finding the outline of the sudoku puzzle in the image
contour_1 = sudoku_a.copy()
contour_2 = sudoku_a.copy()
contour, hierarchy = cv2.findContours(
    threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_1, contour, -1, (0, 255, 0), 3)


def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


def splitcells(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


black_img = np.zeros((450, 450, 3), np.uint8)
biggest, maxArea = main_outline(contour)
plt.show()
if biggest.size != 0:
    biggest = reframe(biggest)
    cv2.drawContours(contour_2, biggest, -1, (0, 255, 0), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    su_imagewrap = cv2.warpPerspective(sudoku_a, matrix, (450, 450))
    su_imagewrap = cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)


sudoku_cell = splitcells(su_imagewrap)


# The sudoku_cell's output includes the boundaries this could lead to misclassifications by the model
# I am cropping the cells to avoid that
# sneeking in a bit of PIL lib as cv2 was giving some weird error that i couldn't ward off

def CropCell(cells):
    Cells_croped = []
    for image in cells:

        img = np.array(image)
        img = img[8:46, 8:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)

    return Cells_croped


sudoku_cell_croped = CropCell(sudoku_cell)


def read_cells(cell, model):

    result = []
    for image in cell:
        # preprocess the image as it was in the model
        img = np.asarray(image)
        # plt.imshow(img, cmap='gray')

        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (dim, dim))
        img = 255 - img

        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, dim, dim, 1)
        # getting predictions and setting the values if probabilities are above 65%

        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        # plt.title(str(classIndex))
        # plt.show()
        probabilityValue = np.amax(predictions)
        #classIndex = np.argmax(predictions,axis=1)

        if probabilityValue > 0.65:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


grid = read_cells(sudoku_cell_croped, clf)
grid = np.asarray(grid)
grid = np.reshape(grid, (9, 9))


# This function finds the next box to solve

def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False

# Function to fill in the possible values by evaluating rows collumns and smaller cells


def possible(quiz, row, col, n):
    #global quiz
    for i in range(0, 9):
        if quiz[row][i] == n and row != i:
            return False
    for i in range(0, 9):
        if quiz[i][col] == n and col != i:
            return False

    row0 = (row)//3
    col0 = (col)//3
    for i in range(row0*3, row0*3 + 3):
        for j in range(col0*3, col0*3 + 3):
            if quiz[i][j] == n and (i, j) != (row, col):
                return False
    return True

# Recursion function to loop over untill a valid answer is found.


def solve(quiz):
    val = next_box(quiz)
    if val is False:
        return True
    else:
        row, col = val
        for n in range(1, 10):  # n is the possible solution
            if possible(quiz, row, col, n):
                quiz[row][col] = n
                if solve(quiz):
                    return True
                else:
                    quiz[row][col] = 0
        return


def Solved(quiz):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("....................")

        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")

            if col == 8:
                print(quiz[row][col])
            else:
                print(str(quiz[row][col]) + " ", end="")


if solve(grid):
    Solved(grid)
else:
    print("Solution don't exist. Model misread digits.")


print('done')


plt.show()
