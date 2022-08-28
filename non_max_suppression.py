# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:11:06 2022

@author: HP
"""
import cv2 
import numpy as np
img=cv2.imread("people.jpg")
#print(img)
#Y'OLO algoritmasında resmin en ve boyuna ihtiyacım var
'''
img.shape
Out[9]: (427, 640, 3) boy en kanal
'''
img_width=img.shape[1]
img_height=img.shape[0]  
''' 1/255 skala değeri YOLO tarafından belirtilmiş en optimal deger
     416 indirdiğimiz modül 
     swapRB RGB'ye çevirme 
     crop resim kırpmak
'''
img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
''' img_blob.shape
Out[13]: (1, 3, 416, 416) 
labels=["person","phone",...] kendi modülümü yamış olsaydı ne kadar nesne
 tanıdığını ve isimleri buraya yazmam gerekiyor'''
#%% Label ve colors ayarlama
labels=["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
colors=np.random.uniform(0,255,size=(len(labels),3))

#%% layers ayarlama

model=cv2.dnn.readNetFromDarknet("pretrained_model/yolov3.cfg", "pretrained_model/yolov3.weights")  #cnfg dosyasını
layers=model.getLayerNames()  #modelden layersleri çekiyorum
#layersta outputları ayıklamak
output_layers=[layers[layer-1] for layer in model.getUnconnectedOutLayers()]   
model.setInput(img_blob)
detection_layers=model.forward(output_layers)


#----------------NMS operation1 start----------------
idsList=[]
boxesList=[]
confidenceList=[]
#----------------NMS operation1 End----------------
#%% 
for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores=object_detection[5:]  #ilk5 değer bounding değerleri tutuyor 5ten sonrakilerle ilgileniyorum
        predicted_id=np.argmax(scores)  #en yüksek score'a sahip olan 
        confidence=scores[predicted_id]  #güvenlik scoru
        
        if confidence > 0.30:
            label=labels[predicted_id]
            #bounding
            bounding_box=object_detection[0:4] * np.array([img_width,img_height,img_width,img_height]) 
            #ilk 4 değer çok küçük ve yeterli olmadığı için genişletiourm
            (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
            start_x=int(box_center_x - (box_width/2))
            start_y=int(box_center_y - (box_height/2))
            #----------------NMS operation2 start----------------
             
            idsList.append(predicted_id)
            confidenceList.append(float(confidence))
            boxesList.append([start_x,start_y,int(box_width),int(box_height)])
            
            #----------------NMS operation2 end----------------
            
#----------------------- NMS Operation3 start---------------
maxids=cv2.dnn.NMSBoxes(boxesList,confidenceList,0.5,0.4)
for maxid in maxids:
    maxClassID=maxid
    box=boxesList[maxClassID]
    start_x=box[0]
    start_y=box[1]
    box_width=box[2]
    box_height=box[3]
    predicted_id=idsList[maxClassID]
    label=labels[predicted_id]
    confidence=confidenceList[maxClassID]
    
#----------------------- NMS Operation3 end---------------
        
    end_x=start_x + box_width
    end_y=start_y + box_height
    
    box_color=colors[predicted_id]
                
                
    label="{}: {:.2f}%".format(label, confidence*100) 
    print("Predicted object {}".format(label)) 
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)
    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
            
cv2.imshow("Detection Window",img)            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


