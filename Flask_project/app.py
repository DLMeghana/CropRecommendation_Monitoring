from flask import Flask,render_template,url_for
import requests
import numpy as np
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
    return render_template('crop-result.html', title=title)

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