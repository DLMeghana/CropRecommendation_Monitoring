from flask import Flask,render_template,request,Markup
import pickle
import os
import cv2
import numpy as np
import pandas as pd
from utils.disease import disease_dic
import requests
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import sklearn
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, redirect, url_for

#--
# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction
#--
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
    title = 'Agro Check - Yield Prediction'

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
    title = 'Agro Check - Yield Prediction'
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
    user_input=[crop_name,season_name,state_name,area,Production,Annual_rainfall,Fertilizer,Pesticide]
    user_input_df=pd.DataFrame(user_input)
    scaler=MinMaxScaler()
    input_data_scaled=scaler.fit_transform(user_input_df)
    input_data_scaled=input_data_scaled.transpose()
    predicted_yield = yield_prediction_model.predict(input_data_scaled)
    result="Predicted yield production:%.3f" %predicted_yield[0][0]
    return render_template('yield-result.html',result=result,title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agro Check - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

@app.route('/weed-predict', methods=['GET', 'POST'])
def weed_detection():
    title = 'Agro Check - Weed Detection'
    return render_template('weed.html', title=title)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    # Ensure the 'uploads' directory exists
    upload_folder = os.path.join(os.getcwd(), 'static/uploads')
    os.makedirs(upload_folder, exist_ok=True)

    # Save the file to the 'uploads' folder
    file.save(os.path.join(upload_folder, file.filename))

    # Redirect to a page that displays the uploaded image
    return redirect(url_for('display_image', filename=file.filename))
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the model architecture
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)

# Load the trained model weights
model.load_state_dict(torch.load('11test_mIou_0.7509999871253967test_loss_0.08900000154972076.pth', map_location=torch.device('cpu')))
model.eval()

# Define class names
names = {'0': 'crop', '1': 'weed'}
import random
import string
@app.route('/weed_result-<filename>')
def display_image(filename):
    image_path = os.path.join(os.path.dirname(__file__), 'static', 'uploads', filename)

    # Check if the file exists
    if not os.path.exists(image_path):
        return 'Image not found'

    # Read the image using OpenCV
    src_img = plt.imread(image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    # Convert image to tensor
    img_tensor = torch.from_numpy(img/255.).permute(2, 0, 1).float()

    # Perform inference
    out = model(torch.unsqueeze(img_tensor, dim=0))

    # Extract bounding boxes, labels, and scores
    boxes = out[0]['boxes'].cpu().detach().numpy().astype(int)
    labels = out[0]['labels'].cpu().detach().numpy()
    scores = out[0]['scores'].cpu().detach().numpy()

    # Convert src_img to cv::UMat object
    src_img_um = cv2.UMat(src_img)

    # Draw bounding boxes and labels
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.8:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(src_img_um, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(src_img_um, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    # Convert src_img_um back to NumPy array for display
    src_img = src_img_um.get()

    # Display the annotated image
    #plt.imshow(src_img)
    #plt.show()
    cv2.imwrite("./static/uploads/prediction.jpeg", src_img)

    #return render_template('weed-result.html', filename="prediction.jpeg")
    random_query_param = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    return render_template('weed-result.html', filename="prediction.jpeg", random_query_param=random_query_param)
# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)