# Shrimp Ponds Information
1. In this task, I wanted to find some information (jumlah kolam, estimasi luas, geohash, geometry polygon, and alamat kolam)
2. I did not have any imagery satellite of ponds images at all
3. I tried to search the location based on the given ponds coordinate from 5 tables on Google Maps. Then, I found the imagery satellite of block of shrimp ponds. Then, I realized that I can use a keyword `shrimp farm` to find other location of shrimp ponds from google maps. I found around 15 or 20 locations around ASEAN, like Indonesia, Thailand, Vietnam, Phillipines, and Myanmar. Then, I take a screen shot of the image from different angle and zoom parameter in order to get variation. As a result, I got around 94 dataset. This dataset will be used to find `jumlah kolam` information. 
4. Then, I tried to search on google to find some information like geohash, geometry polygon, and alamat kolam from given langitude and longitude coordinate. Fortunately, I found the way to get it in python
5. Thus, I used computer vision based object detection approach to get jumlah kolam from given specific langitude and longitude coordinate. I used Google Maps API in order to retrieve the imagery satellite, which will be passed to the object detection model. Then, I used `geohash2`, `geopy`, and `Shapely` to extract some information like alamat kolam, geohash, and alamat kolam

`PS: for the API KEY, I will share it at replying email to Mr.Ardi when I submit this task`

6. For the object detection model, I used EfficientDet as the architecture and CenterNet for the output head. [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) is one of lightweight object detection model and it is very fast to run the inference. Moreover, it has Bi-FPN layer which helps the network to learn detail information of the detected object and its size. So, it will make the model more precise
7. For the output, I used [CenterNet](https://github.com/xingyizhou/CenterNet) because the output is simple. The object tries to predict the center of the object, then regress the bounding box size. CenterNet is one of the object detection model which is anchor-free model. We did not need anchor setting to train it. it consists of three heads (detecting class, bounding box regression, and center-point regression)
8. Training deep neural networks model is time-consuming and expensive. you need more data to make the model robust. But, 38 samples is very little. So, I used augmentation technique to make more variations and amount of data. The augmentation techniques that I used are Blur/Fog, RGBShift, Brightness, Contrast, and ColorJitter, Vertical Flip, Horizontal Flip, and Rotate 90. I got around 1120 images

9. After analyzed the dataset and decided the apporach, we have to annotate the image for the ground truth. The ponds the I annotate or give a bounding box are the ones that fills with water, which color is blue or sea green. The ponds that empty or green (fullfill with grass) are not annotated. To annoate the image, I used [labelimg](https://github.com/heartexlabs/labelImg). Here is the [dataset](https://drive.google.com/drive/folders/10-BFbVzVUPohieRxMBpFcri7jY8StVh-?usp=share_link) of the bounding box with its annotation file

10. Then, we have to split the dataset into three folders (train, and val). The portion are 80%, 20% respectively. 

11. Then, we can train the model. To know the procedure, just check folder

12. Because the training takes sometime, I stop at epoch 300. Then, I evaluate the model and got performance like this with loss 0.46. Based on the AP information below, we can see that we got AP 0.956 if the IoU threshold is 0.5, which is pretty good performance. The mechanism for testing will be explained at folder
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.630
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.956
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.714
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.669
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.024
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
 ```

`PS: I am sorry I can't show the plot of the loss. Because there was an issue to run the tensorboard`

13. After that, I made the inference function to produce the bounding box prediction

14. Then, I made a visualization of detected ponds in the image

15. In order to run it, you can follow `README.md` file. it runs on flask which will open a index.html website. You will be asked to input Name, Island, Langitude, and Longitude coordinate. Then, it will show all of information.

16. For estimasi luas kolam, I failed to find out the approach. So, it is the only information that I did not get

17. I did not implement the part that uses ClearML for experiment tracking and serving because I do not have any knowledge about it. Additionally, with the limited time, I do not have any much time to learn it. I mostly spend my time in data collection, annotation, and data augmentation. And designing the inference pipeline for the final output. 