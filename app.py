from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained models
xgboost_model = joblib.load('xgboost_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    credit_score = float(request.form['CreditScore'])
    age = float(request.form['Age'])
    tenure = float(request.form['Tenure'])
    balance = float(request.form['Balance'])
    num_of_products = float(request.form['NumOfProducts'])
    has_cr_card = float(request.form['HasCrCard'])
    is_active_member = float(request.form['IsActiveMember'])
    estimated_salary = float(request.form['EstimatedSalary'])
    geography = request.form['Geography']
    gender = request.form['Gender']
    
    # Map the selected gender value to the 'gender_male' variable
    gender_male = True if gender == 'Male' else False
    # Map the selected geography value to variables
    geography_germany = True if geography == 'Germany' else False
    geography_spain = True if geography == 'Spain' else False

   

    # Make predictions using the models
    input_features = [credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, geography_germany, geography_spain, gender_male]
    scaled_features = scaler.transform([input_features])
    xgboost_prediction = xgboost_model.predict(scaled_features)[0]
    random_forest_prediction = random_forest_model.predict(scaled_features)[0]
    votes = [xgboost_prediction, random_forest_prediction]
    generate_bar_chart([xgboost_prediction, 1 - xgboost_prediction],'XGboost')  # Assuming xgboost_prediction is a probability
    generate_bar_chart([random_forest_prediction, 1 - random_forest_prediction], 'RandomForest')

    # Determine the final prediction based on majority vote
    if xgboost_prediction > 0:  # If the majority or tie vote is for churn
        result = "Churned"
    else:
        result = "Not Churned"

    if random_forest_prediction > 0:  # If the majority or tie vote is for churn
        result2 = "Churned"
    else:
        result2 = "Not Churned"

    return render_template('result.html', xgboost_prediction=xgboost_prediction, random_forest_prediction=random_forest_prediction, result=result, result2=result2)

def generate_bar_chart(probabilities, name):
    labels = ["Churned", "Not Churned"]
    values = [probabilities[0], probabilities[1]]
    colors = ['#FF5733', '#33FF57']

    plt.figure(figsize=(6, 6))
    plt.bar(labels, values, color=colors)
    plt.title("Prediction Probabilities")
    plt.xlabel("Customer Churn")
    plt.ylabel("Probability")
    plt.ylim(0, 1)

    # Save the chart as an image file
    plt.savefig('static/'+ name +'.png')

if __name__ == '__main__':
    app.run(debug=True)