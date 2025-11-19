import warnings
warnings.filterwarnings('ignore')
import numpy as np
import re
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import string
import pickle
from gensim.models import KeyedVectors
from functools import lru_cache


def preprocess(text_data, decont = False):
    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    sentance = text_data
    for char in string.punctuation:
        sentance = sentance.replace(char, ' ').strip()
    sentance = sentance.replace('\u200d', '')
    if decont:
        sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = sentance.lower().strip()
    return sentance


def create_embedding_matrix(tokenizer, word2vec_model, embedding_size):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_size))
    for word, index in tokenizer.word_index.items():
        if word in word2vec_model:
            embedding_matrix[index] = word2vec_model[word]
    return embedding_matrix

@lru_cache(maxsize=None)
def model_init():
    print("Model Loading")
    def load_glove_vectors(glove_file):
        embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings
    
    english_embeddings = load_glove_vectors("embeddings/glove.6B.300d.txt")

    hindi_embeddings = KeyedVectors.load_word2vec_format("./embeddings/hi.vec", binary=False)

    max_sequence_length = 36

    with open('english_tokenizer.pickle', 'rb') as handle:
        english_tokenizer = pickle.load(handle)
    handle.close()

    with open('hindi_tokenizer.pickle', 'rb') as handle:
        hindi_tokenizer = pickle.load(handle)
    handle.close()

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

    model.load_weights('model_weights.h5')

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

    return encoder_model, decoder_model, english_tokenizer, hindi_tokenizer, max_sequence_length