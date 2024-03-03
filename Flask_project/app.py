from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import sklearn
crop_recommendation_model_path ='models/model.pkl' 
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
yield_prediction_model_path='models/yield_prediction.pkl'
yield_prediction_model=pickle.load(
    open(yield_prediction_model_path,'rb'))
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

@ app.route('/weed-detect')
def Weed_detect():
    title = 'Agro Check - Weed Detection'

    return render_template('weed.html', title=title)


@ app.route('/Yield')
def yield_prediction():
    title = 'Harvestify - Yield Prediction'

    return render_template('yield.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-recommend', methods=['POST'])
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
        result="{}.It is the best crop to be cultivated ".format(crop)
    else:
        result="Sorry we are not able to recommend a proper crop for this environment"
    return render_template('crop-result.html', result=result,title=title)

# render yield-prediction result page


@ app.route('/yield-predict', methods=['POST'])
def yield_predict():
    title = 'Harvestify - Yield Prediction'
    crop_name=str(request.form['cropname'])
    state_name=str(request.form['statename'])
    season_name=str(request.form['seasonname'])
    area=float(request.form['area'])
    Production=float(request.form['production'])
    Annual_rainfall=float(request.form['annual_rainfall'])
    Pesticide=float(request.form['pesticide_amount'])
    Fertilizer=float(request.form['fertilizer_amount'])
    c={'Arecanut':0,'Arhar/Tur':1,'Castor seed':8,'Coconut':9,'Cotton(lint)':11,'Dry chillies':13,'Gram':16,'Jute':21,'Linseed':23,'Maize':24,'Mesta':26,'Niger seed':29,'Onion':31,'Other  Rabi pulses':32,'Potato':37,'Rapeseed &Mustard':39,'Rice':40,'Sesamum':43,'Small millets':44,'Sugarcane':46,'Sweet potato':48,'Tapioca':49,'Tobacco':50,'Turmeric':51,'Wheat':53,'Bajra':2,'Black pepper':5,'Cardamom':6,'Coriander':10,'Garlic':14,'Ginger':15,'Groundnut':17,'Horse-gram':19,'Jowar':20,'Ragi':38,'Cashewnut':7,'Banana':3,'Soyabean':45,'Barley': 4,'Khesari':22,'Masoor':25,'Moong(Green Gram)':27,'Other Kharif pulses':34,'Safflower':41,'Sannhamp':42,'Sunflower':47,'Urad':52,'Peas & beans (Pulses)':36,'other oilseeds':54,'Other Cereals':33,'Cowpea(Lobia)':12,'Oilseeds total':30,'Guar seed':18,'Other Summer Pulses':35,'Moth':28  
}
    s={'Whole Year':4,'Kharif':1,'Rabi':2,'Autumn':0,'Summer':3,'Winter':5}  
    st={'Assam':2,'Karnataka':12,'Kerala':13,'Meghalaya':17,'West Bengal':29,'Puducherry':21,'Goa':6,'Andhra Pradesh':0,'Tamil Nadu':24,'Odisha':20,'Bihar':3,'Gujarat':7,'Madhya Pradesh':14,'Maharashtra':15,'Mizoram':18,'Punjab':22,'Uttar Pradesh':27,'Haryana':8,'Himachal Pradesh':9,'Tripura':26,'Nagaland':19,'Chhattisgarh':4,'Uttarakhand':28,'Jharkhand':11,'Delhi':5,'Manipur':16,'Jammu and Kashmir':10,'Telangana':25,'Arunachal Pradesh':1,'Sikkim':23 } 
    if crop_name in c:
        crop_name=c[crop_name]
    if state_name in st:
        state_name=st[state_name]
    if season_name in s:
        season_name=s[season_name]
    user_input=[area,Production,Annual_rainfall,Fertilizer,Pesticide,crop_name,state_name,season_name]
    user_input_df = pd.DataFrame([user_input])
    predicted_yield = yield_prediction_model.predict(user_input_df)
    result=f"Predicted yield production: {predicted_yield[0][0]}"
    if(predicted_yield[0][0]<0):
        result="Sorry we are unable to predict"

    return render_template('yield-result.html',result=result,title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agro Check - Disease Detection'
    return render_template('disease.html', title=title)

@app.route('/weed-predict', methods=['GET', 'POST'])
def weed_detection():
    title = 'Agro Check - Weed Detection'
    return render_template('weed.html', title=title)



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)