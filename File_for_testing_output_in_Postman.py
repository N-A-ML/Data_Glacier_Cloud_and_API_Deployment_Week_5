#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import joblib
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
import os
app=Flask(__name__)



@app.route('/', methods= ['GET','POST'])
def home():
    if(request.method=="GET"):
        data= "hello world"
        return jsonify({'data':data})
@app.route('/predict/')
def predict():
    model=load('model.joblib')
    sepal_length=request.args.get('sepal_length')
    sepal_width=request.args.get('sepal_width')
    petal_length=request.args.get('petal_length')
    petal_width=request.args.get('petal_width')
    
    test_df= pd.DataFrame({'sepal length':[sepal_length], 'sepal width':[sepal_width], 'petal length':[petal_length], 'petal width':[petal_width]}) 
  
    model_prediction=model.predict(test_df)
    model_prediction=np.around(model_prediction, 2)
    
    output=""
    if model_prediction[0]==0:
        output+="Setosa"
        
    elif model_prediction[0]==1:
        output+="Versicolor"
        
    else:
        output+="Virginica"
          
    
    return jsonify({'Iris Type': str(output)})
if __name__=="__main__":
    app.run(debug=True, 
            use_reloader=False
           )


# In[9]:


#get_ipython().system('jupyter nbconvert Deployment_flask.ipynb --to script')


# In[ ]:




