import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from unidecode import unidecode
from gensim.models import KeyedVectors
tf.config.run_functions_eagerly(True)
from utils import *

data = []
for file in os.listdir('data'):
    tree = ET.parse(f'data/{file}') 
    root = tree.getroot() 
    data += [(x[0].text, x[1].text) for x in root]
data = pd.DataFrame(data, columns = ['English', 'Hindi'])
data['English'] = data['English'].apply(preprocess)
data['English'] = data['English'].apply(unidecode)
data['Hindi'] = data['Hindi'].apply(preprocess)
data1 = data[(data['Hindi'].str.split().apply(len)<=3) & (data['English'].str.split().apply(len)<=3)]

data = []
tree = ET.parse(f'data/NEWS2018_M-EnHi_trn.xml') 
root = tree.getroot() 
data += [(x[0].text, x[1].text) for x in root]
data = pd.DataFrame(data, columns = ['English', 'Hindi'])
data['English'] = data['English'].apply(preprocess)
data['English'] = data['English'].apply(unidecode)
data['Hindi'] = data['Hindi'].apply(preprocess)
data2 = data[(data['Hindi'].str.split().apply(len)<=3) & (data['English'].str.split().apply(len)<=3)]

data = pd.concat([data1, data2], 0)
data.drop_duplicates(inplace = True)

data['Hindi'] = '\t ' + data['Hindi'] + ' \n'

english_embeddings = load_glove_vectors(r"E:/Project_Crown_Grab/embeddings/glove.6B.300d.txt")

hindi_embeddings = KeyedVectors.load_word2vec_format(r"E:\IDM\hi\hi.vec", binary=False)

max_sequence_length = max(data['English'].apply(len).max(), data['Hindi'].apply(len).max())

english_sequences, english_tokenizer = tokenize_and_pad_sequences(data['English'], max_sequence_length)
hindi_sequences, hindi_tokenizer = tokenize_and_pad_sequences(data['Hindi'], max_sequence_length)

target_data = np.zeros_like(hindi_sequences)
target_data[:, 0:max_sequence_length-1] = hindi_sequences[:, 1:max_sequence_length]
target_data = np.expand_dims(target_data, -1)

embedding_size = 300

english_embedding_matrix = create_embedding_matrix(english_tokenizer, english_embeddings, embedding_size)
hindi_embedding_matrix = create_embedding_matrix(hindi_tokenizer, hindi_embeddings, embedding_size)

input_vocab_size = len(english_tokenizer.word_index) + 1
output_vocab_size = len(hindi_tokenizer.word_index) + 1
lstm_size = 256
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(input_vocab_size, embedding_size, mask_zero=True, weights=[english_embedding_matrix],
                              trainable=False)(encoder_inputs)
encoder_dropout = Dropout(0.9)(encoder_embedding)
encoder_lstm = LSTM(lstm_size, return_state=True, return_sequences=True)
_, encoder_state_h, encoder_state_c = encoder_lstm(encoder_dropout)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(max_sequence_length,))
decoder_embedding = Embedding(output_vocab_size, embedding_size, mask_zero=True, weights=[hindi_embedding_matrix],
                              trainable=False)(decoder_inputs)
decoder_dropout = Dropout(0.9)(decoder_embedding)
decoder_lstm = LSTM(lstm_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation='softmax')
outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)

print(model.summary())

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, 
                                                             decay_rate=0.96, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(initial_learning_rate), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 1000
batch_size = 1024
early = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
check = tf.keras.callbacks.ModelCheckpoint('model_weights.h5', save_weights_only = True, save_best_only = True,
                                           verbose = True)
model.fit([english_sequences, hindi_sequences], target_data, epochs=epochs, batch_size=batch_size, validation_split=0.1,
          callbacks=[check, early])

plt.figure(figsize=(8,5))
plt.plot(np.array(model.history.epoch) + 1, model.history.history['loss'], label='train')
plt.plot(np.array(model.history.epoch) + 1, model.history.history['val_loss'], label='val')
plt.legend()

plt.savefig("training_loss_plot.png", dpi=300, bbox_inches='tight')
plt.close() 

model.load_weights('model_weights.h5')

with open('english_tokenizer.pickle', 'wb') as handle:
    pickle.dump(english_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

with open('hindi_tokenizer.pickle', 'wb') as handle:
    pickle.dump(hindi_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

# Encoder model for inference
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model for inference
decoder_state_input_h = Input(shape=(lstm_size,))
decoder_state_input_c = Input(shape=(lstm_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_dropout, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

input_text = "mera naam ismail hai"
output_text = ' '
for x in input_text.split():
    transliterated_text = transliterate(x, encoder_model, decoder_model, english_tokenizer, hindi_tokenizer, max_sequence_length)
    output_text = output_text + ' ' + transliterated_text
print(output_text.strip())
# मेरा नाम इस्माल है

