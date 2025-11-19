# English to Hindi Transliteration
This repo takes english sentence as input and returns a hindi transcripted sentence like “namaste” to “नमस्ते” 

## Evaluation:
- Train accuracy 84% and Validation accuracy 75%.

## Instruction to run:
### You can run this app in two ways app cold start/initializaton time 30-40 sec
1 - As Flask app
- Run the following command in different terminal

<pre>
python fast_app.py
python flask_app.py
</pre>

2 - As FastAPI
- Run the following command in terminal
<pre>
python fast_app.py
</pre>
then 
```python
import requests
url = "http://127.0.0.1:8000/transliterate"

data = {"text": "aur kya hai"}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Transliterated text:", result["output_text"])
else:
    print("Error:", response.status_code, response.text)
```

3 - As python module
```python
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
output_text = transliterate_text('aur kya hai')
```
