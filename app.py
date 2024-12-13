from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Initialize Flask app
app = Flask(__name__)

# Train the model (this can be done outside Flask if the model is pre-trained)
def train_model():
    df = pd.read_csv('Loan_approvals.csv')
    data = df.dropna()
    data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
    data['Dependents'].replace({'3+':4},inplace=True)
    data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

    X = data[['Gender', 'Education', 'ApplicantIncome', 'Credit_History', 'Property_Area']]
    Y = data['Loan_Status']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    return model, sc

# Train model once at the start
model, sc = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        education = int(request.form['education'])
        income = int(request.form['income'])
        credit_history = int(request.form['credit_history'])
        property_area = int(request.form['property_area'])

        # Prepare data for prediction
        input_data = [[gender, education, income, credit_history, property_area]]
        input_data = sc.transform(input_data)  # Standardize the input

        # Predict using the trained model
        prediction = model.predict(input_data)

        # Return result
        if prediction == 1:
            result = "Yes"
        else:
            result = "No"

        return render_template('index.html', prediction_text='Loan Approval Prediction: ' + result)

if __name__ == "__main__":
    app.run(debug=True)
