🎬 Sentiment Analysis using CNN

A deep learning-based sentiment analysis application that classifies movie reviews as **Positive** or **Negative** using a Convolutional Neural Network (CNN).

The project uses the **IMDB Movie Review Dataset**, TensorFlow/Keras for model development, and Streamlit to provide an interactive web interface.

Demo

The application allows users to enter a movie review and receive a sentiment prediction.

Example

**Input:**
> This movie was absolutely amazing and I loved every minute of it.

**Prediction:**
> 🟢 Positive

**Input:**
> This movie was boring, predictable and a complete waste of time.

**Prediction:**
> 🔴 Negative

---

📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone of text.

In this project, a CNN model is trained to classify movie reviews into two categories:

- 🟢 Positive
- 🔴 Negative

The trained model is integrated into a Streamlit application so users can enter their own reviews and get predictions interactively.

Model Architecture

The project uses a CNN for text classification.


Movie Review
     ↓
IMDB Word Encoding
     ↓
Sequence Padding
     ↓
Embedding Layer
     ↓
1D Convolutional Layer
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


**Model configuration**

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Vocabulary Size         |              10,000 |
| Maximum Sequence Length |                 200 |
| Embedding Dimension     |                 128 |
| CNN Filters             |                 128 |
| Kernel Size             |                   5 |
| Dense Layer             |          64 neurons |
| Dropout                 |                 0.5 |
| Optimizer               |                Adam |
| Loss Function           | Binary Crossentropy |
| Maximum Epochs          |                   5 |
| Batch Size              |                  64 |
| Early Stopping          |                 Yes |


**Model Performance**
| Metric             |   Score |
| ------------------ | ------: |
| Test Accuracy      | **89%** |
| Negative Precision |    0.89 |
| Negative Recall    |    0.88 |
| Negative F1-Score  |    0.88 |
| Positive Precision |    0.88 |
| Positive Recall    |    0.89 |
| Positive F1-Score  |    0.89 |

**Confusion Matrix**
                 Predicted
                 Negative  Positive

Actual Negative    11022     1478
Actual Positive     1396    11104

Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn
Streamlit
Natural Language Processing (NLP)
Convolutional Neural Networks (CNN)

⚙️ Installation
1. Clone the repository
git clone https://github.com/Incharainchu-18/sentiment-analysis-cnn.git

2. Navigate to the project
cd sentiment-analysis-cnn

3. Create a virtual environment
python -m venv .venv

4. Activate the environment
**Windows:**

.venv\Scripts\activate

**Linux / macOS / GitHub Codespaces:**

source .venv/bin/activate

**5. Install dependencies**
pip install -r requirements.txt

**6. Run the Application**

Start the Streamlit application:

streamlit run app.py

Then open the local URL displayed in the terminal.

🏋️ Train the Model

To train the CNN model again:

python train.py

The trained model will be saved to:

models/sentiment_cnn_model.keras

The training accuracy plot will be saved to:

plots/training_accuracy.png

📚 Dataset

This project uses the IMDB Large Movie Review Dataset provided through TensorFlow/Keras.

The dataset contains:

25,000 training reviews
25,000 testing reviews
Binary sentiment labels
Positive
Negative

💡 Key Learning Outcomes
Through this project, I worked with:

Natural Language Processing
Text preprocessing
Word indexing and sequence padding
Word embeddings
Convolutional Neural Networks
Binary text classification
Model evaluation
TensorFlow/Keras
Streamlit deployment concepts
Git and GitHub


🔮 Future Improvements
Possible improvements include:

Using more advanced NLP preprocessing
Comparing CNN with LSTM/GRU models
Hyperparameter tuning
Adding model confidence visualization
Deploying the Streamlit application publicly
Experimenting with transformer-based models

👩‍💻 Author

Incharainchu-18

GitHub:
https://github.com/Incharainchu-18

⭐ If you found this project useful

Feel free to explore the repository, experiment with the model, and improve the application.
