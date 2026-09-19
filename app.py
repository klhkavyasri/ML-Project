from flask import Flask, render_template, request
from src.predict import predict_placement

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    prediction = predict_placement(data)

    if prediction == 1:
        result = "Placed"
    else:
        result = "Not Placed"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)