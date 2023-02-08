import json
import pandas as pd
import joblib
import requests
import sys

# input = {
#   "Model": str(input("Model name ")),

#   "HT": {
#            "Mean": float(input("HT Mean ")),
#            "STD": float(input("HT STD "))
#     },

#   "PPT": {
#            "Mean": float(input("PPT Mean ")) ,
#            "STD": float(input("PPT STD "))
#     },

#   "RRT": {
#            "Mean": float(input("RRT Mean ")),
#            "STD": float(input("RRT STD "))
#     },

#   "RPT": {
#            "Mean": float(input("RPT Mean ")),
#            "STD": float(input("RPT STD "))
#     }
# }

# input = {
#     "Model": "SVM",
#     "HT": {
#         "Mean": 48.43,
#         "STD": 23.34
#         },
#     "PPT": {
#         "Mean": 120.43,
#         "STD": 37.41
#     },
#     "RRT": {
#         "Mean": 124.43,
#         "STD": 45.34
#     },
#     "RPT": {
#         "Mean": 132.56,
#         "STD": 47.12
#     }
# }

class Values:
    def __init__(self, model, ht_mean, ht_std, ppt_mean, ppt_std, rrt_mean, rrt_std, rpt_mean, rpt_std):
        self.model = model
        self.ht_mean = ht_mean
        self.ht_std = ht_std
        self.ppt_mean = ppt_mean
        self.ppt_std = ppt_std
        self.rrt_mean = rrt_mean
        self.rrt_std = rrt_std
        self.rpt_mean = rpt_mean
        self.rpt_std = rpt_std

    def print_values(self):
        print(f'{"Model":15}{self.model}')
        print(f'{"HT.Mean":15}{self.ht_mean}')
        print(f'{"HT.STD":15}{ self.ht_std}')
        print(f'{"PPT.Mean":15}{self.ppt_mean}')
        print(f'{"PPT.STD":15}{ self.ppt_std}')
        print(f'{"RRT.Mean":15}{self.rrt_mean}')
        print(f'{"RRT.STD":15}{ self.rrt_std}')
        print(f'{"RPT.Mean":15}{self.rpt_mean}')
        print(f'{"RPT.STD":15}{ self.rpt_std}')


def parse(response):
    try:
        json_object = response.json()
    except requests.exceptions.RequestException:
        print("Bad response format", file=sys.stderr)
        return None
    except:
        print("Unspecified exception while json parsing", file=sys.stderr)
        return None
    try:
        data = json_object["Data"]
    except KeyError:
        print("Field 'Data' not in response", file=sys.stderr)
        return None
    except:
        print("Unspecified exception dictionary key access while in Data", file=sys.stderr)
        return None
    return data


def post_call(url, data_call, headers):
    try:
        response = requests.post(url, data=str(data_call), headers=headers)
    except requests.Timeout:
        print("Timeout error", file=sys.stderr)
        return None
    except requests.ConnectionError:
        print("Connection error", file=sys.stderr)
        return None
    except requests.TooManyRedirects:
        print("Too many redirects error", file=sys.stderr)
        return None
    except:
        print("Undefined Error", file=sys.stderr)
        return None
    return response


def values_post(clienttype=""):
    url = ""
    headers = {"Content-Type": "application/json", "Token": ""}
    data = {"ClientType": str(clienttype),
                   "Data": "null"}

    input = {
    "Model": "SVM",
    "HT": {
        "Mean": 48.43,
        "STD": 23.34
        },
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
    },
    "RRT": {
        "Mean": 124.43,
        "STD": 45.34
    },
    "RPT": {
        "Mean": 132.56,
        "STD": 47.12
    }
    }

    response = post_call(url, data, headers)

    if response:
            data = parse(response, input) 
            if data:

                df = pd.json_normalize(json.loads(data)) 

                if df.iloc[0,0] == 'SVM':
                    loaded_rf = joblib.load("./svm_model.sav")
                    y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
                    print('UserID: ',y_pred.item())

                elif df.iloc[0,0] == 'RF':
                    loaded_rf = joblib.load("./rf_model.sav")
                    y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
                    print('UserID: ',y_pred.item())

                else: 
                    loaded_rf = joblib.load("./xgb_model.sav")
                    y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
                    print('UserID: ',y_pred.item())

            else:
                return None    








# # Converting into JSON:
# output = json.dumps(input)

# # Converting JSON into Dataframe
# df = pd.json_normalize(json.loads(output)) 

# # Making the prediction
# if df.iloc[0,0] == 'SVM':
#     loaded_rf = joblib.load("./svm_model.sav")
#     y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
#     print('UserID: ',y_pred.item())

# elif df.iloc[0,0] == 'RF':
#     loaded_rf = joblib.load("./rf_model.sav")
#     y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
#     print('UserID: ',y_pred.item())

# elif df.iloc[0,0] == 'XGB':
#     loaded_rf = joblib.load("./xgb_model.sav")
#     y_pred = loaded_rf.predict(df.drop(['Model'], axis=1))
#     print('UserID: ',y_pred.item())

# else: print('Wrong model name')



if __name__ == "__main__":
    response = values_post()
    if response:
        response.print_values()
        