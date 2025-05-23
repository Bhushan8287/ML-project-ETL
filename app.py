from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Number_of_Customers_Per_Day = request.form.get('Number_of_Customers_Per_Day'),
            Average_Order_Value = request.form.get('Average_Order_Value'),
            Marketing_Spend_Per_Day = request.form.get('Marketing_Spend_Per_Day')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)