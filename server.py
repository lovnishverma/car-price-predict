from flask import *

import numpy

import pandas as pd

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor


app=Flask(__name__)

#CAR PRICE PREDICTION
@app.route('/')
def carpage():
    return render_template("car.html")
  
@app.route('/Car',methods=["POST"])
def car():

    
  Fueltype=int(request.form.get("fueltype"))
  Enginetype=int(request.form.get("enginetype"))
  Enginesize=eval(request.form.get("enginesize"))
  Horsepower=eval(request.form.get("horsepower"))
  
  data="car.csv"
  data=pd.read_csv(data)
  data=data.values
  x=data[:,[0,1,2,3]]
  y=data[:,-1]
  
  model=DecisionTreeRegressor()
  model.fit(x,y)
  
#   joblib.dump(model,"Car.joblib")
  
# newmodel=joblib.load("Car.joblib")
  
  predict_price=model.predict([[Fueltype,Enginetype,Enginesize,Horsepower]])

  return render_template("car.html",data1=predict_price[0],data2=(predict_price[0])*82.04)


if __name__ == '__main__':
  app.run()