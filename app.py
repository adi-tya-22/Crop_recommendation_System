from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

# Label mapping dictionary
label_mapping = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Make prediction
        prediction = model.predict([[nitrogen, phosphorus, potassium,
                                     temperature, humidity, ph, rainfall]])[0]

        # Crop name
        crop_name = label_mapping[int(prediction)]
        result_text = f"{crop_name.capitalize()} is the best crop to be cultivated right there 🌱"

        # Only send crop name to template
        return render_template("index.html", result=result_text, crop_name=crop_name)

    except Exception as e:
        return render_template("index.html", result=f"❌ Error: {str(e)}", crop_name=None)

if __name__ == "__main__":
    app.run(debug=True)
