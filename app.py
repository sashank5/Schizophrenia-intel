import numpy as np
from flask import Flask, request, jsonify, render_template,request
from chat import get_response
import pickle
from sklearn import preprocessing
#import subprocess

#subprocess.run(["python", "train.py"])

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("home.html")

@flask_app.route("/Prediction", methods = ["GET","POST"])
def prediction():
    return render_template("Prediction.html")

@flask_app.post("/find")
def find():
    text = request.get_json().get("message") 
    response = get_response(text) 
    message = {"answer": response}
    return jsonify(message)     


@flask_app.route("/predict", methods = ["POST"])
def predict():
    form_fields = ['age', 'gender', 'physical_health','mental_health','workpressure','emotionalpressure','mental_ability_workdone','mental_relationship','felt_low','diet_change','last_happy','felt_good','positive','history_disorder','therapist','medication','sleep_quality','smoke','drink']
    new=[]
    for i in form_fields:
        data=request.form[i]
        new.append(data)
    

    features = np.array(new)
    encoder=preprocessing.LabelEncoder()

    features=encoder.fit_transform(features).reshape(1,-1)
   
    prediction = model.predict_proba(features)[0][1]

    occur=prediction*100
    not_occur = (1-prediction)*100
    text="Percentage of chance that you might get affected with Schizophrenia is:", ("%.2f" % occur),"%" 
    text1="Percentage of chance that you might not get affected with Schizophrenia is:", "%.2f" % not_occur,"%"

    child_affect=(0.12)*occur
    text3="Percentage of chance that your child get affected with Schizophrenia is:", ("%.2f" % child_affect),"%" 

    return render_template("Prediction.html", occurance = text,non_occurance=text1,child_occur=text3)
    

    

@flask_app.route("/Prevention", methods = ["GET","POST"])
def prevent():
    return render_template("Prevention.html")



@flask_app.route("/Effects", methods = ["GET","POST"])
def effects():
    return render_template("Effects.html")


@flask_app.route("/Medical-Help", methods = ["GET","POST"])
def medi_help():
    return render_template("Medical-Help.html")


@flask_app.route("/Patient-Experience", methods = ["GET","POST"])
def pati_exp():
    return render_template("Patient-Experience.html")


if __name__ == "__main__":
    flask_app.run(debug=True)