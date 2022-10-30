from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('disease.pkl', 'rb'))

test=pd.read_csv("test_data.csv",error_bad_lines=False)
x_test=test.drop('prognosis',axis=1)


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/FAQ')
def FAQ():
    return render_template('faq.html')
@app.route('/detail')
def detail():
    return render_template('detail.html')
@app.route('/doctor')
def doctors():
    return render_template('doctors.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        col=x_test.columns
        inputt = [str(x) for x in request.form.values()]

        b=[0]*132
        for x in range(0,132):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,132)
        prediction = model.predict(b)
        prediction=prediction[0]
        li=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']
        ans=li[prediction]
    return render_template('index.html', pred="The probable diagnosis says it could be {}".format(ans))







if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)
