from flask import Flask,request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
df=pd.read_csv("campus_placement_dataset.csv")
app = Flask(__name__)

#Deserialize
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html') #due to this function we are able to send our webpage to client(browser) - GET

@app.route('/predict',methods=['POST','GET'])  #gets inputs data from client(browser) to Flask Server - to give to ml model
def predict():
    features = [float(x) for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    #our model was trained on Normalized(scaled) data
    X = df.iloc[:,1:12].values
    sc=StandardScaler().fit(X)
    output = model.predict(sc.transform(final))
    print(output)

    if output[0]==1:
        return render_template('index.html',pred=f'your are placed')
    else:
        return render_template('index.html',pred=f'your are not placed')



if __name__ == '__main__':
    app.run(debug=True)