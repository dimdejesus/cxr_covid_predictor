import pickle
from django.conf import settings

def dict_to_array(dict_data):
    feature_arr = []
    for key, value in dict_data.items():
        feature_arr.append(value)

    return feature_arr

def model_predict(features):
    # No Finding vs Pneumonia
    lda1 = pickle.load(open(f"{settings.MODEL_ROOT}\lda1-nofinding-pneu.sav", "rb"))
    # Other Pneumonia vs COVID-19 Pneumonia
    lda2 = pickle.load(open(f"{settings.MODEL_ROOT}\lda2-covid-pneu.sav", "rb"))

    prediction = lda1.predict([features])[0]

    if(prediction == "Pneumonia"):
        prediction = lda2.predict([features])[0]
        
    return prediction
