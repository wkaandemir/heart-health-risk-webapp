#app.py
import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
model = pickle.load(open("ModelRandomForestClassifier.pkl", "rb"))

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

# Bind predict function to URL
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form entries values
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Create a NumPy array with the input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Predict the risk
        prediction = model.predict(input_data)

        if prediction == 1:
            result = 'Endişelenmenize Gerek Yok. Risk Grubunda Değilsiniz!'
        else:
            result = 'Lütfen Doktorunuza Başvurun!'

        return render_template('index.html', result=result)

    except Exception as e:
        error_message = 'Verileri işlerken bir hata oluştu. Lütfen tüm girişleri doğru bir şekilde doldurun.'
        return render_template('index.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
