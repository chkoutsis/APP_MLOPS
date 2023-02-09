''' Importing dependencies '''
from flask import Flask,render_template,request
import json
import pandas as pd
import joblib
import pathlib

''' Creating the application. '''
app = Flask(__name__)

''' Displays the index page accessible '''
@app.route('/')
def index():
    return render_template('index.html')


''' Creating the prediction function which route the page URL '''
@app.route('/predict',methods=['POST'])
def predict():

    ''' Requesting the JSON format input '''
    f = request.files['json_file']

    ''' Saving the JSON format input '''
    f.save('input.json') 
    
    ''' Reading the JSON format input '''
    with open('input.json', 'r', encoding='utf-8') as file:
        file = json.load(file)

    ''' Converting the JSON into Dataframe '''
    df = pd.json_normalize(file) 
    
    ''' Deleting the JSON file from the folder '''
    file = pathlib.Path('input.json')
    file.unlink()

    ''' Making the prediction '''
    if df.iloc[0,0] == 'SVM':
        loaded_rf = joblib.load("./models/svm_model.joblib")
        y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
        return render_template('index.html',result=y_pred.item())
    elif df.iloc[0,0] == 'RF':
        loaded_rf = joblib.load("./models/rf_model.joblib")
        y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
        return render_template('index.html',result=y_pred.item())
    elif df.iloc[0,0] == 'XGB':
        loaded_rf = joblib.load("./models/xgb_model.joblib")
        y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
        return render_template('index.html',result=y_pred.item())
    else: return render_template('index.html',result='')
    

if __name__ == '__main__':
    app.run(debug=True)