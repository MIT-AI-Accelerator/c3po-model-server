from threading import Lock
import math
import tensorflow as tf
import pandas as pd
import os
from app.core.logging import logger
from sample_data import CHAT_DATASET_1_PATH

BASE_CKPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "training_checkpoints")
def initialize():

    # load the dataset
    training_data_path = CHAT_DATASET_1_PATH
    chat_data_path = training_data_path if os.path.isfile(training_data_path) else os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data_open/example_data.csv")
    chat816 = pd.read_csv(chat_data_path)

    # Make a text-only dataset (without labels), then call adapt
    train_text = tf.constant(chat816['messages'].astype(str).values)

    # number of unique words in the dataset after punctuation filtering
    word_count_layer = tf.keras.layers.TextVectorization()
    word_count_layer.adapt(train_text)
    num_words = len(word_count_layer.get_vocabulary())
    logger.info(f'There are {num_words} unique words in this dataset')

    # tokenizer
    # https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
    # https://www.tensorflow.org/tutorials/keras/text_classification
    max_features = math.floor(num_words * .25)
    sequence_length = 25

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        ngrams=3,
        output_sequence_length=sequence_length)

    # build vocab for this dataset
    vectorize_layer.adapt(train_text)

    # save the vocabulary as a standard python list
    vocab = vectorize_layer.get_vocabulary()

    # how far did ngram truly go
    max_ngram_size = 0
    for item in vocab:
        max_ngram_size = max(max_ngram_size, len(item.split()))

    return (vocab, vectorize_layer)


# define the LSTM
def lstm(rnn_units, stateful=True):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=stateful,
    )

### Defining the RNN Model ###


def build_model(vocab_size, num_class, embedding_dim, rnn_units, batch_size):

    first_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True) if batch_size is None else tf.keras.layers.Embedding(
        vocab_size, embedding_dim, batch_input_shape=[batch_size, None], mask_zero=True)

    lstm_layer = lstm(
        rnn_units, stateful=False) if batch_size is None else lstm(rnn_units)

    model_out = tf.keras.Sequential([

        # Layer 0: mask zeros in time steps, i.e., data does not exist
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
        # https://www.tensorflow.org/guide/keras/masking_and_padding
        # tf.keras.layers.Masking(mask_value=0.0),

        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        # mask zeros for diff length inputs
        first_layer,

        # dropout to prevent overfitting
        tf.keras.layers.Dropout(.2),

        # Layer 2: LSTM with `rnn_units` number of units.
        lstm_layer,

        # dropout to prevent overfitting
        tf.keras.layers.Dropout(.2),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size. NOTE: output will need to have softmax applied...no activation
        tf.keras.layers.Dense(num_class)
    ])

    return model_out


#################################################################
# Hyperparameters relevant for inference (make sure they match
# those used in training) ###
# Model parameters:
num_classes = 3
embedding_dim_in = 8
rnn_units_in = 128  # Experiment between 1 and 2048

# # Checkpoint location:
checkpoint_dir = BASE_CKPT_DIR
############## Inference ######################

class LstmStressClassifierModel:
    def __init__(self):
        # use to manage concurrent requests
        self.lock = Lock()

        # initialize the model without loading weights
        self.refresh_model()

    def refresh_model(self):
        self.vocab, self.vectorize_layer = initialize()

        # batch size None for inference, remove statefulness and allow any size input
        self.model_def = build_model(vocab_size=len(
            self.vocab), num_class=num_classes, embedding_dim=embedding_dim_in, rnn_units=rnn_units_in, batch_size=None)

        # Restore the model weights for the last checkpoint after training
        self.model_def.build(tf.TensorShape([1, None]))

    def load_weights(self):
        self.model_def.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    ## Prediction of a chat class ###
    def classify(self, chats):
        with self.lock:
            # convert to list if needed
            chats = [chats] if isinstance(chats, str) else chats

            # up the dimension if single inference...its hacky i know...
            single_item = False
            if len(chats) == 1:
                chats.append("")
                single_item = True

            # Evaluation step (generating ABC text using the learned RNN model)
            input_eval = self.vectorize_layer(tf.squeeze(chats))
            pred = self.model_def(input_eval, training=False)
            pred = tf.nn.softmax(tf.squeeze(pred)[:, -1, :])
            output_labels = tf.argmax(pred, axis=1)

            return output_labels if not single_item else [output_labels[0]]

    def classify_label(self, chats):
        if not isinstance(chats, (list, str)):
            return [""]

        # record if invalid type in array & make it possible to pass to model
        if isinstance(chats, list):
            chats = [chat if isinstance(chat, str) else "" for chat in chats]

        encoded_labels = self.classify(chats)
        labels = ['recycle', 'review', 'action']

        # convert to label name and filter if given None or invalid type in the array
        output = ["" if chats[index] ==
                  "" else labels[label] for index, label in enumerate(encoded_labels)]

        return output

    def classify_single_label(self, chat):
        if not isinstance(chat, (str)):
            return ""

        return self.classify_label([chat])[0]
