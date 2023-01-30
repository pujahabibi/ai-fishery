# Document OCR
1. In this task, I want to extract all of information in the form document of daily cultivation
2. I only got 38 sample images for training data
3. After observed all of samples, I got some information that of which 27 entities that I need to extract (tanggal, blok, kolam, DOC, jenis pakan, f_d, frekuensi pakan jam 7, frekuensi pakan jam 11, frekuensi pakan jam 15, frekuensi pakan jam 20, ANCHO, DO pagi, DO sore/malam, pH pagi, ph sore/malam, suhu pagi, suhu sore/malam, SAL, mati, T air, KEC, warna, cuaca, siphon, kincir, treatment air)
4. only 1 entity that I did not extract which is TDS. because, based on 38 samples, there is no sample where the TDS column have a content
5. For this approach, we have 2 stages. first stage is to detect all of entity. Then, at the second stage, all of detected entity will be passed to `pytesseract` library to convert text-image to string. 
5. I decided to use object detection apporach
6. The method is, I trained an object detection architecture to detect and localize the 27 enitity. Because, since the document is static, which means only have one template or layout, it will be easy for object detection to explicitly memorize the layout. 
7. I used EfficientDet as the architecture and CenterNet for the output head. EfficientDet is one of lightweight object detection model and it is very fast to run the inference. Moreover, it has Bi-FPN layer which helps the network to learn detail information of the detected object and its size. So, it will make the model more precise
8. For the output, I used CenterNet because the output is simple. The object tries to predict the center of the object, then regress the bounding box size. CenterNet is one of the object detection model which is anchor-free model. We did not need anchor setting to train it. it consists of three heads (detecting class, bounding box regression, and center-point regression)
9. Training deep neural networks model is time-consuming and expensive. you need more data to make the model robust. But, 38 samples is very little. So, I used augmentation technique to make more variations and amount of data. The augmentation techniques that I used are Blur/Fog, RGBShift, Brightness, Contrast, and ColorJitter. The reasons why I did not use geometric augmentation (Vertical Flip or Rotation) is because we want the model to learn the structure or the layout of the document. If we use geometric augmentation, it will make the model hard to recognize the layout. 

here is some of sample of augmentation image
brightness
![](write/0ECD0408-7711-4241-8CA1-7168B339C92E_brightness.JPG) 