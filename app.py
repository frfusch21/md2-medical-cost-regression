from flask import Flask, render_template, request, redirect, make_response
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/predict", methods=['GET', 'POST'])
def pred():
    df = pd.DataFrame(columns=['age','sex','bmi','children','smoker','region'])

    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        sex = request.form["sex"]
        bmi = request.form["bmi"]
        children = request.form["children"]
        smoker = request.form["smoker"]
        region = request.form["region"]

        df = df.append({'age': age, 'sex': sex, 'bmi': bmi, 'children': children,'smoker': smoker, 'region': region}, ignore_index=True)
        with open('model/Lin_reg_model.pkl','rb') as file:
            Lin_reg_model = pickle.load(file)

        prediction = Lin_reg_model.predict(df)

        with open('data_collection.txt','a') as file:
            file.write("%s\n" % df)
    
        return render_template('result.html', pred=str(prediction[0]), name=name)

if __name__ == "__main__":
    app.debug=True
    app.run()