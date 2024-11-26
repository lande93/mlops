from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import numpy as np

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Use the new names from the HTML form
        data = CustomData(
            feat_0=float(request.form.get('feat_0')),
            feat_1=float(request.form.get('feat_1')),
            feat_2=float(request.form.get('feat_2')),
            feat_3=float(request.form.get('feat_3')),
            feat_4=float(request.form.get('feat_4')),
            feat_5=float(request.form.get('feat_5')),
            feat_6=float(request.form.get('feat_6')),
            feat_7=float(request.form.get('feat_7')),
            feat_8=float(request.form.get('feat_8')),
            feat_9=float(request.form.get('feat_9')),
            feat_10=float(request.form.get('feat_10')),
            feat_11=float(request.form.get('feat_11')),
            feat_12=float(request.form.get('feat_12')),
            feat_13=float(request.form.get('feat_13')),
            feat_14=float(request.form.get('feat_14')),
            feat_15=float(request.form.get('feat_15')),
            feat_16=float(request.form.get('feat_16')),
            feat_17=float(request.form.get('feat_17')),
            feat_18=float(request.form.get('feat_18')),
            feat_19=float(request.form.get('feat_19'))
        )
        pred_data = data.get_data_as_data_frame()
        print(pred_data)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("During Prediction")
        results = predict_pipeline.predict(pred_data)
        print("After Prediction")
        
        # Interpreting the model output
        if results[0] == 1:
            message = "There are high chances that customer will default. Immediate attention required."
        else:
            message = "There are no chances of customer default. It is performing well for now."
        
        return render_template('home.html', results=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
