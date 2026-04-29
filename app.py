import streamlit as st
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from utils import encode_review

# Load Model
model = load_model('models/sentiment_cnn_model.keras')

# Streamlit Page Config
st.set_page_config(
    page_title='Sentiment Analysis CNN',
    page_icon='🎬',
    layout='centered'
)

# Title
st.title('🎬 Sentiment Analysis using CNN')

st.write('Predict whether a movie review is Positive or Negative')

# Text Area
review = st.text_area('Enter Movie Review')

# Button
if st.button('Analyze Sentiment'):

    if review.strip() == '':
        st.warning('Please enter a review')

    else:

        # Encode Review
        processed_review = encode_review(review)

        # Prediction
        prediction = model.predict(processed_review)

        score = prediction[0][0]

        # Positive
        if score > 0.5:
            sentiment = 'Positive 😊'
            confidence = score * 100
            st.success(sentiment)

        # Negative
        else:
            sentiment = 'Negative 😞'
            confidence = (1 - score) * 100
            st.error(sentiment)

        st.write(f'Confidence Score: {confidence:.2f}%')

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=confidence,
            title={'text': 'Confidence'},
            gauge={
                'axis': {'range': [0, 100]}
            }
        ))

        st.plotly_chart(fig)