from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from cxr_covid_predictor.image_processing.image_preprocessing import auto_masking
from cxr_covid_predictor.image_processing.feature_extraction import feature_extract_chunks
from cxr_covid_predictor.image_processing.model_prediction import dict_to_array, model_predict

import os

def media_flush():
    #deleting file to avoid memory full
    files = os.listdir(settings.MEDIA_ROOT)
    for file in files:
        os.remove(os.path.join(settings.MEDIA_ROOT, file))

def index(request):
    media_flush()
    return render(request, 'index.html')

def predict(request):
    if(request.method == "POST") and request.FILES['xray']:
        upload = request.FILES['xray']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)

        #temporary file path
        img_file_path = str(request.FILES['xray'].temporary_file_path()) 
        
        #get the masked_img corresponding to the mask
        masked_img, mask = auto_masking(img_file_path)

        #get its features
        img_features = feature_extract_chunks(masked_img, mask)

        #transforming dict to array for model prediction convenience
        feature_arr = dict_to_array(img_features)

        #print(img_features)
        #print()
        #print(feature_arr)
        #print(len(feature_arr))
        #print()
        #print(type(feature_arr))

        predicted = model_predict(feature_arr)
        print(predicted)

        data = {"result": predicted, "img_file": file_url}

        return render(request, 'result.html', data)
    media_flush()
    return HttpResponse("Please go back to Homepage")