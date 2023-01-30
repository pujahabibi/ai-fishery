# Document OCR
1. In this task, I want to extract all of information in the form document of daily cultivation
2. I only have 38 given images for training dataset
3. After observed all of samples, I got some information that of which 27 entities that I need to extract (tanggal, blok, kolam, DOC, jenis pakan, f_d, frekuensi pakan jam 7, frekuensi pakan jam 11, frekuensi pakan jam 15, frekuensi pakan jam 20, ANCHO, DO pagi, DO sore/malam, pH pagi, ph sore/malam, suhu pagi, suhu sore/malam, SAL, mati, T air, KEC, warna, cuaca, siphon, kincir, treatment air)
4. only 1 entity that I did not extract which is TDS. because, based on 38 samples, there is no sample where the TDS column have a content
5. For this approach, we have 2 stages. first stage is to detect all of entity. Then, at the second stage, all of detected entity will be passed to `pytesseract` library to convert text-image to string. 
6. I decided to use object detection apporach
7. The method is, I trained an object detection architecture to detect and localize the 27 enitity. Because, since the document is static, which means only have one template or layout, it will be easy for object detection to explicitly memorize the layout. 
8. I used EfficientDet as the architecture and CenterNet for the output head. [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) is one of lightweight object detection model and it is very fast to run the inference. Moreover, it has Bi-FPN layer which helps the network to learn detail information of the detected object and its size. So, it will make the model more precise
9. For the output, I used [CenterNet](https://github.com/xingyizhou/CenterNet) because the output is simple. The object tries to predict the center of the object, then regress the bounding box size. CenterNet is one of the object detection model which is anchor-free model. We did not need anchor setting to train it. it consists of three heads (detecting class, bounding box regression, and center-point regression)
10. Training deep neural networks model is time-consuming and expensive. you need more data to make the model robust. But, 38 samples is very little. So, I used augmentation technique to make more variations and amount of data. The augmentation techniques that I used are Blur/Fog, RGBShift, Brightness, Contrast, and ColorJitter. The reasons why I did not use geometric augmentation (Vertical Flip or Rotation) is because we want the model to learn the structure or the layout of the document. If we use geometric augmentation, it will make the model hard to recognize the layout. Here is the [sample](https://drive.google.com/drive/folders/1DKcrJBi6hnW3HtHClzK7vYzV9koIiOVy?usp=share_link) of the augmented data. I got around 322 dataset

11. After analyzed the dataset and decided the apporach, we have to annotate the image for the ground truth, which consist of class and bounding box. To annoate the image, I used [labelimg](https://github.com/heartexlabs/labelImg). Here is the [dataset](https://drive.google.com/file/d/1KspFMjJnvqt0QIAY9ANuGwqNBXcfzNlD/view?usp=share_link) of the bounding box with its annotation file

12. Then, we have to split the dataset into three folders (train and val). The portion are 80%, 20%, respectively. 

13. Then, we can train the model. To know the procedure, just check in README.md at CenterNet folder

14. Because the training takes sometime, I stop at epoch 600. Then, I evaluate the model and got performance like this with loss 0.37. Based on the AP information below, we can see that we got AP 0.932 if the IoU threshold is 0.5, which is pretty good performance. The mechanism for testing will be explained in README.md at CenterNet folder
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=800 ] = 0.662
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=800 ] = 0.932
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=800 ] = 0.775
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=800 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=800 ] = 0.661
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=800 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.180
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.709
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=800 ] = 0.709
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=800 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=800 ] = 0.706
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=800 ] = 0.698
 ```

`PS: I am sorry I can't show the plot of the loss. Because there was an issue to run the tensorboard`

15. After that, I made the inference function to produce the bounding box prediction

16. Then, I integrated with `pytesseract` to read the text from the image. I did not make the text reader model from scratch because I had no time to train it with limited time. 

17. The limitations of using `pytesseract` is the model is bad at reading handwritten text. Moreoever, the crop image of the text is a little bit blurry which made the model hard to predict the text and give bad performance.

18. Finally, the next step that I wanna do is to do post-processing technique in order to produce the output format like the table. However, I did not have time to finish it

19. I did not implement the part that uses ClearML for experiment tracking and serving because I do not have any knowledge about it. Additionally, with the limited time, I do not have any much time to learn it. I mostly spend my time in data collection, annotation, and data augmentation. And designing the inference pipeline for the final output. 