import requests
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        url = "http://127.0.0.1:8000/transliterate"
        data = {"text": text}

        response = requests.post(url, json=data, timeout=360)

        if response.status_code == 200:
            result = response.json()
            output_text = result["output_text"].strip()
            return render_template("index.html", output_text=output_text)
        else:
            error = f"Error: {response.status_code} {response.text}"
            return render_template("index.html", error=error)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

