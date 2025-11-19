import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        try:
            sampled_char = output_tokenizer.index_word[sampled_token_index]
        except:
            sampled_char = ' '

        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_sequence_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()