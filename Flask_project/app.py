from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas
import sklearn
crop_recommendation_model_path ='models/SVMClassifier.pkl' 
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
app=Flask(__name__)
@ app.route('/')
def home():
    title = 'Agro Check - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Agro Check - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page

@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)


@ app.route('/Yield')
def yield_prediction():
    title = 'Harvestify - Yield Prediction'

    return render_template('yield.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    T=float(request.form['temperature'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    Humidity = float(request.form['humidity'])
    data = [N, P, K, T, Humidity, ph, rainfall]
    single_pred= np.array(data).reshape(1,-1)
    prediction=crop_recommendation_model.predict(single_pred)

    crop_dict=["Rice","Maize","Jute","Cotton","Coconut","Papaya","Orange","Apple","Maskmelon","Watermelon","Grapes","Mango","Banana","Pomegranate","Lentil","Blackgram","Mungbean","Mothbeans","Pigeonpeas","Kidneybeans","Chickpea","Coffee"]
    if prediction[0].title() in crop_dict:
        crop=prediction[0].title()
        result="{} is a best crop to be cultivated ".format(crop)
    else:
        result="Sorry we are not able to recommend a proper crop for this environment"
    return render_template('crop-result.html', result=result,title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer-result.html',title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    return render_template('disease.html', title=title)

@app.route('/weed-predict', methods=['GET', 'POST'])
def weed_detection():
    title = 'Harvestify - Weed Detection'
    return render_template('weed.html', title=title)



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)