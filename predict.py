from tensorflow.keras.models import load_model
from utils import encode_review

# Load Model
model = load_model('models/sentiment_cnn_model.keras')

# User Input
review = input('Enter Movie Review: ')

# Preprocess
processed_review = encode_review(review)

# Predict
prediction = model.predict(processed_review)

score = prediction[0][0]

# Display Result
if score > 0.5:
    print(f'Positive Review 😊 ({score:.2f})')
else:
    print(f'Negative Review 😞 ({score:.2f})')