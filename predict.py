from utils import *
from trans import transliterate

encoder_model, decoder_model, english_tokenizer, hindi_tokenizer, max_sequence_length = model_init()
def transliterate_text(text):
    
    text = preprocess(text)
    text = '\t ' + text + ' \n'

    output_text = ' '
    for x in text.split():
        transliterated_text = transliterate(x, encoder_model, decoder_model, english_tokenizer, hindi_tokenizer, max_sequence_length)
        output_text = output_text + ' ' + transliterated_text

    return output_text.strip()
