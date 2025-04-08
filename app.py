from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('youtube_revenue_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        views = float(request.form['views'])
        likes = float(request.form['likes'])
        shares = float(request.form['shares'])
        watch_time = float(request.form['watch_time'])
        subscribers = float(request.form['subscribers'])
        subscribers_gained = float(request.form['subscribers_gained'])
        dislikes = float(request.form['dislikes'])
        average_percentage_viewed = float(request.form['average_percentage_viewed'])

        # Prepare data for prediction
        input_data = np.array([[views, likes, shares, watch_time, subscribers, subscribers_gained, dislikes, average_percentage_viewed]])

        # Prediction
        prediction = model.predict(input_data)
        predicted_usd = round(prediction[0], 2)
        
        # Conversion to INR
        conversion_rate = 86 # Customize this based on current forex
        predicted_inr = round(predicted_usd * conversion_rate, 2)
        
        return render_template('result.html', usd=predicted_usd, inr=predicted_inr)

if __name__ == "__main__":
    app.run(debug=True)
