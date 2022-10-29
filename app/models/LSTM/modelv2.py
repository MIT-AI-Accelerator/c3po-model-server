# Import Tensorflow 2.0
import enum
import tensorflow as tf
from tensorflow.keras import layers
import math

# Import all remaining packages
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
assert len(tf.config.list_physical_devices('GPU')) > 0

# load the dataset
chat816 = pd.read_csv(
    '../../../data/(EROB) MM_Dataset_816_CSVsanitized_flights.csv')

# Make a text-only dataset (without labels), then call adapt
train_text = tf.constant(chat816['Column12'].astype(str).values)

# number of unique words in the dataset after punctuation filtering
word_count_layer = layers.TextVectorization()
word_count_layer.adapt(train_text)
num_words = len(word_count_layer.get_vocabulary())
print(f'There are {num_words} unique words in this dataset')

# tokenizer
# https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
# https://www.tensorflow.org/tutorials/keras/text_classification
max_features = math.floor(num_words * .25)
sequence_length = 25

vectorize_layer = layers.TextVectorization(
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


print(f'The maximum ngram found was {max_ngram_size}.')

# # see some vocab details
print(" 10 ---> ", vocab[10])
print(" 50 ---> ", vocab[50])
print(" -1 ---> ", vocab[-1])
print(f'Vocabulary size: {len(vocab)}')

# def vectorize_text(text, label):
#   text = tf.expand_dims(text, -1)
#   return vectorize_layer(text), label

# vectorize the input
vector_chats = vectorize_layer(train_text)

# vectorize the output
vector_target = (chat816.loc[:, ['recycle', 'review', 'action']].astype(
    str) == 'x').astype(float).values

### Batch definition to create training examples ###


def get_batch(vectorized_chats, vectorized_target, batch_size):

    # number of chats
    n = vectorized_chats.shape[0]

    # randomly choose the starting indices for the examples in the training batch
    sample_indices = np.random.choice(n, batch_size)

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = tf.constant(vectorized_chats[sample_indices, :])
    y_batch = tf.constant(vectorized_target[sample_indices, :])

    return x_batch, y_batch

# define the LSTM


def LSTM(rnn_units, stateful=True):
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

    LSTM_layer = LSTM(rnn_units, stateful=False) if batch_size is None else LSTM(rnn_units)

    model = tf.keras.Sequential([

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
        LSTM_layer,

        # dropout to prevent overfitting
        tf.keras.layers.Dropout(.2),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size. NOTE: output will need to have softmax applied...no activation
        tf.keras.layers.Dense(num_class)
    ])

    return model

# loss function, negative log likelihood


def compute_loss(labels, logits):
    # from_logits means we compare against the output probability distribution
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    return loss


#################################################################
### Hyperparameter setting and optimization ###
# Optimization parameters:
num_training_iterations = 20001  # Increase this to train longer
batch_size = 32  # Experiment between 1 and 64
initial_learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
num_classes = 3
embedding_dim = 8
rnn_units = 128  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

### Define optimizer and training operation ###
# Build a simple model with above hyperparameters.
model = build_model(vocab_size=len(vocab), num_class=num_classes,
                    embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)

# learning rate schedule
# see here:https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=7500,
    decay_rate=0.1,
    staircase=True)


# specific gradient descent algorithm choice.
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
)


@tf.function
def train_step(x, y):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:

        # generate predictions. Use training=True since we have dropout (acts differently in tng vs inference,
        # see here: https://www.tensorflow.org/tutorials/quickstart/advanced)
        y_hat = model(x, training=True)

        # get the input label
        y_sample = tf.math.argmax(y, axis=1, output_type=tf.dtypes.int64)
        # tf.print(y, summarize=-1)
        # tf.print(y_sample, summarize=-1)

        loss = compute_loss(y_sample, y_hat[:, -1, :])

        # Compute the gradients
        # We want the gradient of the loss with respect to all of the model parameters.
        # Use `model.trainable_variables` to get a list of all model parameters.
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the optimizer so it can update the model accordingly
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

##################
# Begin training!#
##################

# history = []
# # plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
# plt.title('Loss over time')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

# for iter in tqdm(range(num_training_iterations)):

#   # Grab a batch and propagate it through the network
#   x_batch, y_batch = get_batch(vector_chats.numpy(), vector_target, batch_size=batch_size)
#   loss = train_step(x_batch, y_batch)

# #   print("Input shape:      ", x_batch.shape, " # (batch_size, sequence_length)")
# #   print("Prediction shape: ", y_batch.shape, "# (batch_size, sequence_length, vocab_size)")

#   # Update the progress bar
#   history.append(loss.numpy().mean())

#   # Update the model with the changed weights!
#   if iter % 100 == 0:
#     model.save_weights(checkpoint_prefix)
#     plt.plot(range(iter + 1), history, 'g', label='Training loss')
#     plt.show()

# # Save the trained model and the weights
# model.save_weights(checkpoint_prefix)
# # plt.plot(epochs, loss_val, 'b', label='validation loss')    plt.show()
# plt.plot(range(101,num_training_iterations), history[101:num_training_iterations], 'g', label='Training loss')


############## Inference ######################


# batch size None for inference, remove statefulness and allow any size input
model = build_model(vocab_size=len(vocab), num_class=num_classes, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=None)

# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

## Prediction of a chat class ###
def classify(chats):

    # convert to list if needed
    chats = [chats] if isinstance(chats, str) else chats

    # up the dimension if single inference...its hacky i know...
    single_item = False
    if len(chats) == 1:
       chats.append("")
       single_item = True

    # Evaluation step (generating ABC text using the learned RNN model)
    input_eval = vectorize_layer(tf.squeeze(chats))
    pred = model(input_eval)
    pred = tf.nn.softmax(tf.squeeze(pred)[:, -1, :])
    output_labels = tf.argmax(pred, axis=1)

    return output_labels if not single_item else [output_labels[0]]


def classify_label(chats):
    encoded_labels = classify(chats)
    labels = ['recycle', 'review', 'action']
    return list(map(lambda label: labels[label], encoded_labels))

# calculate accuracy
pred_labels = classify(train_text)
accuracy = np.sum(np.equal(
    list(map(np.argmax, vector_target)), pred_labels))/len(train_text)
print(f'Accuracy is {accuracy}')