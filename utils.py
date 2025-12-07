import warnings
warnings.filterwarnings('ignore')
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
tf.config.run_functions_eagerly(True)

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

def load_glove_vectors(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def tokenize_and_pad_sequences(texts, max_sequence_length):
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences, tokenizer

def create_embedding_matrix(tokenizer, word2vec_model, embedding_size):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_size))
    for word, index in tokenizer.word_index.items():
        if word in word2vec_model:
            embedding_matrix[index] = word2vec_model[word]
    return embedding_matrix

def transliterate(input_text, encoder_model, decoder_model, input_tokenizer, output_tokenizer, max_sequence_length):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['\t']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = output_tokenizer.index_word[sampled_token_index]

        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_sequence_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()




