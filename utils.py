from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dataset Settings
vocab_size = 10000
max_length = 200

# Word Index
word_index = imdb.get_word_index()


def encode_review(text):
    words = text.lower().split()

    encoded = []

    for word in words:
        # IMDB reserves special token indices
        index = word_index.get(word, 2)

        # Convert raw IMDB word index to the same format
        # used by imdb.load_data()
        index += 3

        # Unknown/out-of-vocabulary words
        if index >= vocab_size:
            index = 2

        encoded.append(index)

    padded = pad_sequences(
        [encoded],
        maxlen=max_length
    )

    return padded
