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

        """ feature_arr = [70.4308943089431, 
                        893.562297574195, 
                        0.0575993675253721, 
                        111, 
                        2.35613833213081, 
                        64.3720812182741, 
                        1034.87323069391, 
                        -0.130694489027529, 
                        138, 
                        2.00103172444059, 
                        54.738278516445, 
                        815.872719499755, 
                        0.299952844334358, 
                        133, 
                        1.91637249569665, 
                        56.8900169204737, 
                        1192.64611015199, 
                        0.0790996594291557, 
                        126, 
                        1.64732870028379, 
                        45.8519715578539, 
                        940.468714600354, 
                        0.683372060977436, 
                        123, 
                        1.49515369290662, 
                        35.4163742690058, 
                        808.374000889162, 
                        1.12869439940775, 
                        121, 
                        1.2456554425443, 
                        42.0428249436513, 
                        1019.81108863877, 
                        0.649914866685631, 
                        126, 
                        1.31653383370683, 
                        37.4292993630573, 
                        987.371116069617, 
                        0.967369699243306, 
                        125, 
                        1.19116380512535, 
                        55.9621040723981, 
                        357.702749419033, 
                        1.05234293733231, 
                        104, 
                        2.95891776740091, 
                        69.3232916972814, 
                        461.184240744449, 
                        0.296200104844327, 
                        103, 
                        3.22806259151898, 
                        66.8050420168067, 
                        990.3451846621, 
                        0.12423022749362, 
                        110, 
                        2.12283355780545, 
                        46.1408740359897, 
                        615.226426999557, 
                        1.17945845298935, 
                        113, 
                        1.86023720410096, 
                        71.3803596127247, 
                        548.938314116875, 
                        1.0431929567544, 
                        105, 
                        3.04661067517093, 
                        97.054054054054, 
                        352.591672753834, 
                        -0.379818836180096, 
                        45, 
                        5.16865631878698, 
                        0, 
                        0, 
                        0, 
                        0, 
                        0, 
                        62.0026200873362, 
                        1284.37291889933, 
                        0.686087329772101, 
                        140, 
                        1.7300731743798, 
        ] """

        predicted = model_predict(feature_arr)

        data = {"result": predicted, "img_file": file_url}

        return render(request, 'result.html', data)
    media_flush()
    return HttpResponse("Please go back to Homepage")