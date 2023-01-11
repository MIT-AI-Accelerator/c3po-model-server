# Import Tensorflow 2.0
import enum
import tensorflow as tf

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
chat816 = pd.read_csv('../../../data/(EROB) MM_Dataset_816_CSVsanitized_flights.csv')

# print(chat816.iloc[0:1,:])

# # Find all words used in chats
# vocab_array = []
# max_length = 0
# for chat in chat816['Column12'].astype(str):
#     row_array = chat.split()
#     if len(row_array) > max_length:
#         max_length = len(row_array)

#     for word in row_array:
#         vocab_array.append(word)

# # Find all unique words in the chats
# vocab_array.append("\n\n")
# vocab = sorted(set(vocab_array))
# print("There are", len(vocab), "unique words in the dataset")

# tokenizer 2
# https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/

# Find all characters in chats
all_text = "\n\n".join(chat816['Column12'].astype(str).values.tolist())
max_length = 0
for chat in chat816['Column12'].astype(str):
    if len(chat) > max_length:
        max_length = len(chat)

vocab = sorted(set(all_text))
print("There are", len(vocab), "unique characters in the dataset")
print("The max chat length is ", max_length)

# print(list(vocab)[0:200])
# print(vocab_array[0:20])

# build lookup tables for conversion from word to index
# ensure '0' not used as that is for masking
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
# https://www.tensorflow.org/guide/keras/masking_and_padding
word2index = {word:index + 1 for index, word in enumerate(vocab)}
word2index['MASKED'] = 0

idx2word = {index + 1:word for index, word in enumerate(vocab)}
idx2word[0] = 'MASKED'

#vectorize the chats
num_chats = len(chat816['Column12'])
vector_chats = np.zeros(shape = (num_chats, max_length))
for i, chat in enumerate(chat816['Column12'].astype(str)) :
    # chat_array = chat.split()
    chat_array = chat

    for j in range(len(chat_array)):
        vector_chats[i, j] = word2index[chat_array[j]]

# print(vector_chats[0,:])
# print(chat816['Column12'][0])

# print(vector_chats.shape[0])
# print(len(vector_chats))

#vectorize the output
vector_target = (chat816.loc[:,['recycle','review','action']].astype(str) == 'x').astype(float).values

# labels only, argmax
# vector_target_labels_only = (chat816.loc[:,['recycle','review','action']].astype(str) == 'x').astype(float).apply(pd.Series.argmax, axis=1).values

### Batch definition to create training examples ###
def get_batch(vectorized_chats, vectorized_target, batch_size):

  # number of chats
  n = vectorized_chats.shape[0]

  # randomly choose the starting indices for the examples in the training batch
  sample_indices = np.random.choice(n,batch_size)

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = tf.constant(vectorized_chats[sample_indices, :])
  y_batch = tf.constant(vectorized_target[sample_indices, :])

  return x_batch, y_batch

# define the LSTM
def LSTM(rnn_units):
  return tf.keras.layers.LSTM(
    rnn_units,
    return_sequences=True,
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

### Defining the RNN Model ###
def build_model(vocab_size, num_class, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([

    # Layer 0: mask zeros in time steps, i.e., data does not exist
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
    # https://www.tensorflow.org/guide/keras/masking_and_padding
    # tf.keras.layers.Masking(mask_value=0.0),

    # Layer 1: Embedding layer to transform indices into dense vectors
    #   of a fixed embedding size
    # mask zeros for diff length inputs
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None], mask_zero=True),

    # Layer 2: LSTM with `rnn_units` number of units.
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size.
    tf.keras.layers.Dense(num_class, activation='sigmoid')
  ])

  return model

# sample from categorical distro
def sampleFromCategorical(categoricalTensor):
    # Note that this is log probs, see here https://www.tensorflow.org/api_docs/python/tf/random/categorical
    return tf.random.categorical(categoricalTensor, num_samples=1)

# loss function, negative log likelihood
def compute_loss(labels, logits):
    # from_logits means we compare against the output probability distribution
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  return loss

#################################################################
### Hyperparameter setting and optimization ###
# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 32  # Experiment between 1 and 64
initial_learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
num_classes = 3
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

### Define optimizer and training operation ###
# Build a simple model with above hyperparameters.
model = build_model(vocab_size=len(vocab), num_class=num_classes, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)

# learning rate schedule
# see here:https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=750,
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

    # generate predictions
    y_hat = model(x)

    # get the input label
    y_sample = tf.math.argmax(y,axis=1, output_type=tf.dtypes.int64)
    # tf.print(y, summarize=-1)
    # tf.print(y_sample, summarize=-1)

    loss = compute_loss(y_sample, y_hat[:,-1,:])

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

history = []
# plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
plt.title('Loss over time')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for iter in tqdm(range(num_training_iterations)):

  # Grab a batch and propagate it through the network
  x_batch, y_batch = get_batch(vector_chats, vector_target, batch_size=batch_size)
  loss = train_step(x_batch, y_batch)

#   print("Input shape:      ", x_batch.shape, " # (batch_size, sequence_length)")
#   print("Prediction shape: ", y_batch.shape, "# (batch_size, sequence_length, vocab_size)")

  # Update the progress bar
  history.append(loss.numpy().mean())

  # Update the model with the changed weights!
  if iter % 100 == 0:
    model.save_weights(checkpoint_prefix)
    plt.plot(range(iter + 1), history, 'g', label='Training loss')
    plt.show()

# Save the trained model and the weights
model.save_weights(checkpoint_prefix)
# plt.plot(epochs, loss_val, 'b', label='validation loss')    plt.show()


############## Inference ######################


# # batch size 1 for inference
# model = build_model(vocab_size=len(vocab), num_class=num_classes, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=1)

# # Restore the model weights for the last checkpoint after training
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# model.build(tf.TensorShape([1, None]))

# model.summary()

### Prediction of a chat class ###

# def classify(model, chat):
#   # Evaluation step (generating ABC text using the learned RNN model)
#   input_eval = ['''TODO''']
#   input_eval = tf.expand_dims(input_eval, 0)

#   # Empty string to store our results
#   text_generated = []

#   # Here batch size == 1
#   model.reset_states()
#   tqdm._instances.clear()

#   for i in tqdm(range(generation_length)):
#       '''TODO: evaluate the inputs and generate the next character predictions'''
#       predictions = model('''TODO''')

#       # Remove the batch dimension
#       predictions = tf.squeeze(predictions, 0)

#       '''TODO: use a multinomial distribution to sample'''
#       predicted_id = tf.random.categorical('''TODO''', num_samples=1)[-1,0].numpy()

#       # Pass the prediction along with the previous hidden state
#       #   as the next inputs to the model
#       input_eval = tf.expand_dims([predicted_id], 0)

#       '''TODO: add the predicted character to the generated text!'''
#       # Hint: consider what format the prediction is in vs. the output
#       text_generated.append('''TODO''')

#   return (start_string + ''.join(text_generated))



# x_batchr, y_batchr = get_batch(vector_chats, vector_target, batch_size=batch_size)
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# pred = model(x_batchr)
# print(pred[0:15,50,:])
# print(y_batchr[0:15])
# print(x_batchr[0])
