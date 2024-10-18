# Military-Aircraft-Detection-with-YOLOv8

<img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/Su57.png"><img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/F117-1.png">

<img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/Mirage2000-Rafale.png"><img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/V22.png">

<img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/MQ9-1.png"><img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/MQ9-2.png">

<img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/F35-2.png"><img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/F14.png">

<img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/B52.png"><img src="https://github.com/em-1001/Military-Aircraft-Detection-with-YOLOv8/blob/master/aircraft%20image/C130-2.png">

### False Positive(wrong class)Permalink
**F22, F22, F35, F35 -> F35, F35, F35, F35**  
**F14, F18, Rafale, F15 -> F14, F18, F15, F16**  
**Su57 -> F22**  

### NMS(Non-maximum Suppression)
|Detection|YOLOv8n|YOLOv8s|YOLOv8m|YOLOv8l|YOLOv8x|
|-|-|-|-|-|-|
|v8 + MSE + CA|?|?|8.47|?|?|
|v8 + CIoU + CA|?|?|8.48|?|?|
|v8 + SIoU + CA|?|?|?|?|?|

# How to train 


# Reference 
## Web Link 
One-stage object detection : https://machinethink.net/blog/object-detection/  
YOLOv5 : https://blog.roboflow.com/yolov5-improvements-and-evaluation/   
YOLOv8 : https://blog.roboflow.com/whats-new-in-yolov8/       
mAP : https://blog.roboflow.com/mean-average-precision/    
SiLU : https://tae-jun.tistory.com/10     
Weight Decay, BN : https://blog.janestreet.com/l2-regularization-and-batch-norm/  
Focal Loss : https://gaussian37.github.io/dl-concept-focal_loss/  
　　　　 　https://woochan-autobiography.tistory.com/929  
Cross Entropy : https://sosoeasy.tistory.com/351  
DIOU, CIOU : https://hongl.tistory.com/215    
QFL, DFL : https://pajamacoder.tistory.com/m/74  
YOLOv8 Pytorch : https://github.com/jahongir7174/YOLOv8-pt  

## Paper 
Real-Time Flying Object Detection with YOLOv8 : https://arxiv.org/pdf/2305.09972.pdf   
YOLO : https://arxiv.org/pdf/1506.02640.pdf    
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf    
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
YOLOv4 : https://arxiv.org/pdf/2004.10934.pdf   
YOLOv6 : https://arxiv.org/pdf/2209.02976.pdf  
YOLOv7 : https://arxiv.org/pdf/2207.02696.pdf   
CIoU : https://arxiv.org/abs/1911.08287    
QFL, DFL, GFL : https://arxiv.org/abs/2006.04388   
