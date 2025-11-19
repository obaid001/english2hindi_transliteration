from fastapi import FastAPI
from pydantic import BaseModel
from utils import *
from trans import transliterate
import time

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post('/transliterate')
def transliterate_text(data: InputData):
    srt = time.time()
    encoder_model, decoder_model, english_tokenizer, hindi_tokenizer, max_sequence_length = model_init()
    print(time.time() - srt)

    text = data.text

    # Preprocess the input text
    text = preprocess(text)
    text = '\t ' + text + ' \n'

    output_text = ' '
    for x in text.split():
        transliterated_text = transliterate(x, encoder_model, decoder_model, english_tokenizer, hindi_tokenizer, max_sequence_length)
        output_text = output_text + ' ' + transliterated_text

    return {"output_text": output_text.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
