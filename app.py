from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__, template_folder='templates')

@app.route('/', methods = ['GET'])
def home():
    return render_template('/home.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('/index.html')
    else:
        form_data_list = []
        for field_name, field_value in request.form.items():
            form_data_list.append(field_value)

        test_df = np.array(form_data_list).reshape(1,-1)
        print(test_df.shape)
        # print(form_data_list)
        # test_df = np.array([0.721274551521743,0.16914058806845,0.390284354359258,0.000326977269203,0.0007250725072507,0.370594257300249,0.004094405952288,0.302646433889668,0.147949938898487,0.0212659243655332]).reshape(1,-1)
        
        pipeline = joblib.load('pipeline.pkl')
        
        features = ['WorkingCapital/Equity',
                    'PersistentEPSintheLastFourSeasons',
                    'BorrowingDependency',
                    'NetValueGrowthRate',
                    'InterestBearingDebtInterestRate',
                    'ROA(C)BeforeInterestAndDepreciationBeforeInterest',
                    'Cash/TotalAssets',
                    'NonIndustryIncomeAndExpenditure/Revenue',
                    'NetValuePerShare(B)',
                    'TotalDebt/TotalNetWorth']
        final_X_train = pipeline.named_steps['scaler'].transform(test_df)
        final_X_train = pd.DataFrame(final_X_train,columns=features)
        prediction = pipeline.named_steps['model'].predict(final_X_train)
        result = ''
        if prediction[0] == 1:
            result = 'Bankruptcy'
        else:
            result = 'Not Bankruptcy'

        # test = [[0.721274551521743,0.16914058806845,0.390284354359258,0.000326977269203,0.0007250725072507,0.370594257300249,0.004094405952288,0.302646433889668,0.147949938898487,0.0212659243655332],[0.728730796471968,0.161482461945731,0.384998982291879,0.0003517569706697,0.0008050805080508,0.390922829425243,0.0229885979786101,0.302814414633032,0.158821794277527,0.0244412223346921]]

        # test_df = np.array(test)
        # # print(test_df.shape)
        
        return  render_template('/index.html', prediction_text = 'The company is {}'.format(result))
        # return jsonify({'prediction': "Bankruptcy"})

if __name__ == '__main__':
    app.run(debug=True)