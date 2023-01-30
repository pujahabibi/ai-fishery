# Vibrio Detection
1. In this task, I wanted to detect and count the number of vibrio colony based on their color from the microscope image. 
2. I only have 20 given images for training dataset

3. Because the output is to count all detected vibrio colony, it is obvious that it should use object detection approach

4. For the object detection model, I used EfficientDet as the architecture and CenterNet for the output head. [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) is one of lightweight object detection model and it is very fast to run the inference. Moreover, it has Bi-FPN layer which helps the network to learn detail information of the detected object and its size. So, it will make the model more precise
5. For the output, I used [CenterNet](https://github.com/xingyizhou/CenterNet) because the output is simple. The object tries to predict the center of the object, then regress the bounding box size. CenterNet is one of the object detection model which is anchor-free model. We did not need anchor setting to train it. it consists of three heads (detecting class, bounding box regression, and center-point regression)
6. Training deep neural networks model is time-consuming and expensive. you need more data to make the model robust. But, 20 samples is very little. So, I used augmentation technique to make more variations and amount of data. The augmentation techniques that I used are Blur/Fog, RGBShift, Brightness, Contrast, and ColorJitter, Vertical Flip, Horizontal Flip, and Rotate 90. I got around 176 images

7. Then I observed that from 20 images, it only consists of 2 colors, which is yellow and green. The challenge is that there are some of vibrios that attach together, which is very hard to count how many of them. In order to determine the number of vibrio that attach one another, I look at how many 'circle side' of the vibrio. 

8. After analyzed the dataset and decided the apporach, we have to annotate the image for the ground truth. To annoate the image, I used [labelimg](https://github.com/heartexlabs/labelImg). Here is the [dataset](https://drive.google.com/drive/folders/10-BFbVzVUPohieRxMBpFcri7jY8StVh-?usp=share_link) of the bounding box with its annotation file

9. Then, we have to split the dataset into three folders (train, and val). The portion are 80%, 20% respectively. 

10. Then, we can train the model. To know the procedure, just check in README.md at CenterNet folder

11. Because the training takes sometime, I stop at epoch 350. Then, I evaluate the model and got performance like this with loss 0.46. Based on the AP information below, we can see that we got AP 0.752 if the IoU threshold is 0.5, which is pretty good performance. The mechanism for testing will be explained in README.md at CenterNet folder
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.752
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.031
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.247
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 ```

`PS: I am sorry I can't show the plot of the loss. Because there was an issue to run the tensorboard`

12. After that, I made the inference function to produce the bounding box prediction

13. Then, I made a visualization of detected vibrio of certain color in the image. Also, I display the output counting. 

14. In order to run it, you can follow `README.md` file. it runs on flask which will open a index.html website. You will be asked to upload the image of vibrio from microscope image.

15. I did not implement the part that uses ClearML for experiment tracking and serving because I do not have any knowledge about it. Additionally, with the limited time, I do not have any much time to learn it. I mostly spend my time in data collection, annotation, and data augmentation. And designing the inference pipeline for the final output. 