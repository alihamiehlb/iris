from flask import Flask, render_template, request
import numpy as np
import pickle
import gunicorn


# Load the trained model
model = pickle.load(open("iris_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    # Ensure inputs are in the correct format (2D array)
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction
    predictions = model.predict(input_features)

    # Map predictions to flower names
    flower_class = {0: "setosa", 1: "versicolor", 2: "virginica"}
    result = flower_class[predictions[0]]

   
    image_map = {
        "setosa": "static/setosa.jpeg",
        "versicolor": "static/versicolor.jpg",
        "virginica": "static/verginica.jpg"  
    }
    flower_image = image_map[result]

    return render_template('index.html', prediction_text=f"The predicted flower is {result}", flower_image=flower_image)

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host="0.0.0.0",port=5000)
