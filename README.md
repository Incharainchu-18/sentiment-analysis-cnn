Sentiment Analysis using CNN

A deep-learning based NLP application that classifies movie reviews as **Positive** or **Negative** using a 1D Convolutional Neural Network (CNN).

Overview

This project uses the **IMDB movie review dataset** provided by TensorFlow/Keras to train a binary sentiment classifier.

The trained model is exposed through:

-  Streamlit web application
-  Command-line prediction
-  Training accuracy visualization
-  Confidence score visualization

Model Architecture

text
Movie Review
    ↓
IMDB Word Encoding
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
1D Convolution
    ↓
Global Max Pooling
    ↓
Dense Layer
    ↓
Dropout
    ↓
Sigmoid Output
    ↓
Positive / Negative
