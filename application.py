import pickle
from flask import Flask, request, jsonify, render_template


application = Flask(__name__)
app = application

## Import the models 
model = pickle.load(open('models/log_reg_model.pkl', 'rb'))
scaler = pickle.load(open('models/stdscaler.pkl', 'rb'))

@ app.route('/')
def index():
    return render_template('index.html')

@ app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=='POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin',0))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        pred = model.predict(new_data)

        if pred == 1:
            result = "Daibetic"
        else:
            result = "Non-Daibetic"

        return render_template('pred.html', result=result)
    else:
        return render_template('pred.html') 

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000)