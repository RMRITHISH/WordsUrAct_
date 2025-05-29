import cv2
import numpy as np
import pandas as pd
import math
import sys
import os
import torch
from torchvision import transforms
import pyttsx3 
from wordsegment import load,segment
import enchant
import autocomplete
from gtts import gTTS
from translate import Translator
from playsound import playsound
from Run import MyModel  # Assumes MyModel is defined in Run.py

def process():
    num=0
    engine = pyttsx3.init() 
    cap = cv2.VideoCapture(0)
    img_width = 1000
    img_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    ll=[]
    def image_resize(image, height = 45, inter = cv2.INTER_AREA):
        resized = cv2.resize(image, (height,height), interpolation = inter)
        return resized

    num_classes = 35  # Set this to match your trained model
    model = MyModel(num_classes)
    model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
    model.eval()

    encoding_chart = pd.read_csv('label_encoded.csv')
    encoding_values = encoding_chart['Encoded'].values
    encoding_labels = encoding_chart['Label'].values
    int_to_label = dict(zip(encoding_values,encoding_labels))

    font = cv2.FONT_HERSHEY_DUPLEX

    history = list()
    counts = dict()
    history_length = 15
    threshold = 0.9

    start = 200
    end = 500
    alpha = 0.4

    sentence_raw = list()
    color = (59, 185, 246)

    load()
    sente=""
    disp=""
    sent=""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    while(True):
        ret, img = cap.read()
        img = cv2.flip(img,1)
        alpha_layer = img.copy()
        source = img.copy()

        crop_img = source[start:end, start:end]
        cv2.circle(alpha_layer, (int((start+end)/2),int((start+end)/2)), int((end - start)/2), color ,-1)
        cv2.addWeighted(alpha_layer, alpha, img, 1 - alpha,0, img)

        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized = image_resize(crop_img)
        input_tensor = transform(resized).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
        predicted_char = int_to_label[predicted.item()]
        cv2.putText(img,"Legend:",(490,160),font,1,(255,255,255),1)
        cv2.putText(img,"k-Kannada Speech",(490,190),font,1,(255,255,255),1)
        cv2.putText(img,"t-Tamil Speech",(490,220),font,1,(255,255,255),1)
        cv2.putText(img,"s-Telugu Speech",(490,250),font,1,(255,255,255),1)
        cv2.putText(img,"h-Hindi Speech",(490,280),font,1,(255,255,255),1)
        cv2.putText(img,"a-append",(490,310),font,1,(255,255,255),1)
        cv2.putText(img,"p-space",(490,340),font,1,(255,255,255),1)
        cv2.putText(img,"x-Delete",(490,370),font,1,(255,255,255),1)
        cv2.putText(img,"X-Clear",(490,400),font,1,(255,255,255),1)
        
        if(len(history)>=history_length):
            keys = list(counts.keys())
            values = list(counts.values())
            arg = np.argmax(values)
            if(values[arg]>threshold*history_length):
                sentence_raw.append(keys[arg])
            counts.clear()
            history.clear()
        if(predicted_char != 'None'):
            history.append(predicted_char)
            if(predicted_char in counts):
                counts[predicted_char]+=1
            else:
                counts[predicted_char]=1
            textsize = cv2.getTextSize(predicted_char, font, 6,7)[0]
            textX = int(start + ((end - start) - textsize[0])/2)
            textY = int(end - ((end - start) - textsize[1])/2)
            cv2.putText(img, predicted_char, (textX,textY),font,6,color,7)
        scribble = "".join(sentence_raw)
        sentence = " ".join(segment(scribble))
        
        k=cv2.waitKey(1)
        
        if k==ord('a'):
            sente+=predicted_char
        elif k==ord('p'):
            sente+=" "
        
        cv2.putText(img,sente,(80,50),font,1,(255,255,255),2)
        
        if k == ord('x'):
            sente=sente[:len(sente)-1]
        elif k==ord('X'):
            sente=""
        elif k==ord('k'):
            translator = Translator()
            from_lang = 'en'
            to_lang='kn'
            text_to_translate = translator.translate(sente,src= from_lang,dest= to_lang)
            sente1=text_to_translate.text
            output = gTTS(text=sente1, lang = to_lang, slow = False)
            output.save("outputk.mp3")
            playsound("outputk.mp3")
            os.remove("outputk.mp3")
        elif k==ord('t'):
            translator = Translator()
            from_lang = 'en'
            to_lang='ta'
            text_to_translate = translator.translate(sente,src= from_lang,dest= to_lang)
            sente1=text_to_translate.text
            output = gTTS(text=sente1, lang = to_lang, slow = False)
            output.save("outputt.mp3")
            playsound("outputt.mp3")
            os.remove("outputt.mp3")
        elif k==ord('s'):
            translator = Translator()
            from_lang = 'en'
            to_lang='te'
            text_to_translate = translator.translate(sente,src= from_lang,dest= to_lang)
            sente1=text_to_translate.text
            output = gTTS(text=sente1, lang = to_lang, slow = False)
            output.save("outputa.mp3")
            playsound("outputa.mp3")
            os.remove("outputa.mp3")
        elif k==ord('h'):
            translator = Translator()
            from_lang = 'en'
            to_lang='hi'
            text_to_translate = translator.translate(sente,src= from_lang,dest= to_lang)
            sente1=text_to_translate.text
            output = gTTS(text=sente1, lang = to_lang, slow = False)
            output.save("outputh.mp3")
            playsound("outputh.mp3")
            os.remove("outputh.mp3")
       
        if k == 27:
            break
        
        d=enchant.Dict("en_US")
        try:
            disp=""
            sent1=sente.split()
            a=d.suggest(str(sent1[-1]))
            for cc,tt in enumerate(a):
                if cc==10:
                    break
                disp+=str(cc)+str(".")+str(tt)+" "
        except:
            kkk=0
        cv2.putText(img,disp,(0,120),font,1,(255,255,255),1)
        cv2.putText(img,"Suggestions:",(0,90),font,1,(255,255,255),1)
        cv2.putText(img,"Text:",(0,50),font,1,(255,255,255),1)
        
        if k==48:
           sent=sente.split()
           if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[0])
           else:
               sente=str(a[0])
        elif k==49:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[1]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[1])
            else:
               sente=str(a[1])
        elif k==50:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[2]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[2])
            else:
               sente=str(a[2])
        elif k==51:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[3])
            else:
               sente=str(a[3])
        elif k==52:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[4])
            else:
               sente=str(a[4])
        elif k==53:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[5])
            else:
               sente=str(a[5])
        elif k==54:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[6])
            else:
               sente=str(a[6])
        elif k==55:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[7])
            else:
               sente=str(a[7])
        elif k==56:
            sent=sente.split()
            if len(sent)>1:
               lastword=a[0]
               sente=""
               for gg in range(len(sent)-1):
                   sente+=str(sent[gg])+" "
               sente+=str(a[8])
            else:
               sente=str(a[8])
        cv2.imshow('WebCam', img)
#process()

    
    
