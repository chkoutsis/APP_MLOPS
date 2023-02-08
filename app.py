from flask import Flask,render_template,request
import json
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    # Requesting JSON file
    request_json = request.form.get('json_file')
    with open(request_json, 'r') as f:
        json_file = json.load(f)

    # Converting JSON into Dataframe
    df = pd.json_normalize(json_file)

    # Making the prediction
    if df.iloc[0,0] == 'SVM':
        loaded_rf = joblib.load("./models/svm_model.sav")
        y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
        print('UserID: ',y_pred.item())
        return render_template('index.html',result=y_pred.item())

    elif df.iloc[0,0] == 'RF':
        loaded_rf = joblib.load("./models/rf_model.sav")
        y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
        print('UserID: ',y_pred.item())
        return render_template('index.html',result=y_pred.item())

    elif df.iloc[0,0] == 'XGB':
        loaded_rf = joblib.load("./models/xgb_model.sav")
        y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
        print('UserID: ',y_pred.item())
        return render_template('index.html',result=y_pred.item())

    else: print('Wrong model name')
    return render_template('index.html',result='')



if __name__ == '__main__':
    app.run(debug=True)