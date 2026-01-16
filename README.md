# MEyes
Mobile eyes, an app to support visually impaired by processing what they can't see through the eyes of their wearables and mobile devices.

## Navigation
Path and obstacle detection to help find the best way forward.
Will use a 3 step approach.

### Where can i go?
This will be determined by a segmentation model to define possible paths and obstacles.
After research and discussion with LLM the selected model is LRASPP-MobileNetV3 on pytorch, this is a model that is considered good for outdoor approach and with potential for indoor walkable path detection.
Model still requires fine-tuning for both indoor and outdoor, follow its progression https://github.com/brunoclpinto/LRASPP_MobileNetV3_Walkable

### What's coming my way
Detects potential collisions by tracking object movements and whether their path intersects with ours.
For this YOLOv8+ n|s models will be taken into consideration depending on processing speed.

### STOP NOW
A fail safe depth detection system that is meant to trigger when the previous 2 fails to notice impending obstacle using FastDepth.

## Bus
Yep, detect if bus is comming and try to process its number/destination information.
A 2 step process this time.

### Bus is comming!!! 
Detects bus is on route to your location using a Yolo model, which will also be responsible for detecting and separating the image part that contains bus information.

### Read the info
Reads the bus information for you, so you can know if this is the one you've been wainting for.
Apple Vision Text Recognition (VNRecognizeTextRequest) or SVTR-Tiny / SVTR-Small.

## Read buttons and device screens
Goal is to be able to read button options on a device, elevator or whatever.
Should be able to read all options and single out options by pointing finger. 