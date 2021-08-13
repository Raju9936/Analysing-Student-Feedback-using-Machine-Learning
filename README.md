# Analysing-Student-Feedback-using-Machine-Learning
Machine Learning methods like GNB, DecisionTrees, RandomForest were analysed on student feedback data set


## Table of Contents

1. [Project Description](#Project-description)
2. [Directory Layout](#directory-layout)
3. [set up](#set-up)
4. [Data Set](#data-set)
5. [Loading Libraries](#loading-libraries)
6. [Data Preprocessing and Preparation](#Data-Preprocessing-and-preprocessing)

## Project Description

The project is about the analysing the senitment of the student feedback using the supervised machine learning algorithms using python programming language. Information about user’s sentiment is used for a variety of purposes, such as determining their opinion, attitude towards a business or a product. Whereas with the students’ sentiments it can be used to address issues such as learning experience, teaching, and evaluation etc. Analysing sentiment from the Textual feedback manually is a tedious task and require a lot a of time. This project proposes the methods for analysing sentiment in the student feedback using supervised machine learning models such as Decision Trees, Gaussian Na¨ıve Bayes (GNB), Random Forest. This project analysis the sentiment feedback data set and finds the error metrices like Accuracy, Precision, Recall, f1-Score.
- You can understand about the project clearly from the table of contents which gives details about libraries, installation steps, Data set etc.
- If any one have any queries about the project message me in twitter: ![Twitter](https://img.shields.io/twitter/follow/lenin46685519?style=social)

## Directory Layout

```
- README.md ----> This file you are reading which has all the instructions and clear explanation of the project.
- images ---> This file contains all the images used in the readme file.
- StudentFeedbackML.ipynb ---> This file contains the actual code of the project and detailed description about each block.
- feedback dataset.csv ---> This folder contains the dataset used for the project.
```

## Set up

This project is done using the Jupyter notebook which is preinstalled in anaconda software. The Anaconda can be found here: [Anaconda](https://www.anaconda.com/products/individual) According to the system requirements download 64-bit or 32-bit windows version and it can be downloaded to Mac and Linux systems.
- The required packages are installing tensor flow
```py
pip install tensorflow

```
- After this the jupyter note book is ready with for running machine learning python code.

## Data Set
- The Data set is extracted from the kaggle open source 
(Source: https://www.kaggle.com/chandusrujan/sentimental-analysis-on-student-feedback retrieved in july 2021.)
- The Data set can be found in our repository : [Data Set](https://github.com/Raju9936/Analysing-Student-Feedback-using-Machine-Learning/blob/main/feedback%20dataset.csv)
- The Data set consists of 2 coloums reviews and the sentiment either 1 or 0
- The Data set consists of 5200 rows and it is balanced data set with almost equal number of postive and negative reviews

## Loading Libraries 

```py
import warnings
import numpy as np #Importing the necessary numeric data packages and data analysis packages
import pandas as pd
import re #Regural expression module
import nltk #Natural language toolkit
from nltk.corpus import stopwords #corpus is large and structured set of text
from string import punctuation #loading set of punctuations from string library
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score# importing the error metrices from sklearn library
from sklearn.naive_bayes import GaussianNB#importing the Gaussian Naive bayes algorithm 
import seaborn as sns#importing seaborn library for graphics
%matplotlib inline 
from matplotlib import pyplot as ply
from sklearn.tree import DecisionTreeClassifier#importing Decision tree classifier algorithm
from sklearn.ensemble import RandomForestClassifier #importing the Random forest algorithm 
```

## Data Pre-processing and preparation


