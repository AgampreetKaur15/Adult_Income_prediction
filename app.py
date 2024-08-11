from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=float(request.form.get('age')),
            fnlwgt=float(request.form.get('fnlwgt')),
            education_num=float(request.form.get('education_num')),
            capital_gain=float(request.form.get('capital_gain')),
            capital_loss=float(request.form.get('capital_loss')),
            hours_per_week=float(request.form.get('hours_per_week')),
            workclass=request.form.get('workclass'),
            education=request.form.get('education'),
            marital_status=request.form.get('marital_status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            sex=request.form.get('sex'),
            country=request.form.get('country')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictionPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction", results)

        salary_prediction = 'greater than 50k' if results[0] == 1 else 'less than 50k'
        return render_template('home.html', results=salary_prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8030, debug=True)
