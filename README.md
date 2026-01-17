# MEyes
Mobile eyes, an app to support visually impaired by processing what they can't see through the eyes of their wearables and mobile devices.

## Where will it work
Well, since models can be converted and logic can be translated into different languages, lets go and say everywhere.
But being honest, will first start with IOS, followed by Android cause (FGS rulezzz) and might create alternatives for wifi cams on macos, windows and linux at later stage.
I know, going nuts with this, bare in mind that it will be slow and most likely not steady progress.

## Functions
### Navigation
Path and obstacle detection to help find the best way forward.
Will use a 3 step approach.

#### Where can i go?
This will be determined by a segmentation model to define possible paths and obstacles.
After research and discussion with LLM the selected model is LRASPP-MobileNetV3 (BiSeNet-Cityscapes also under consideration) on pytorch, this is a model that is considered good for outdoor approach and with potential for indoor walkable path detection.
Model still requires fine-tuning for both indoor and outdoor, follow its progression https://github.com/brunoclpinto/LRASPP_MobileNetV3_Walkable

#### What's coming my way
Detects potential collisions by tracking object movements and whether their path intersects with ours.
YOLO will handle this one, will most likely be trained for this app specific needs, check it out https://github.com/brunoclpinto/YOLO-MEyes

#### STOP NOW
A fail safe depth detection system that is meant to trigger when the previous 2 fail to notice impending obstacle using FastDepth.

#### Matrix mapping
Convert segmentation into 2D viable path matrix where fastast path can be easily calculated.
Maybe i'll make it work, would be awesome though!!

### Bus
Yep, detect if bus is comming and try to process its number/destination information.
A 2 step process this time.

#### Bus is comming!!! 
Detects bus is on route to your location using a YOLO, which will also be responsible for detecting, so i can separate the image part that contains, bus information for OCR.

#### Read the info
Reads the bus information for you, so you can know if this is the one you've been wainting for.
Apple Vision Text Recognition (VNRecognizeTextRequest) or SVTR-Tiny / SVTR-Small.
Most likely gonna go native for IOS.

### Read buttons and device screens
Goal is to be able to read button options on a device, elevator or whatever.
Should be able to read all options and single my choice by pointing finger.
Believe this one is gonna be fun to make it work, we'll see.

## Working on
### Bus
First simple task, detect bus, get the info, let me know its comming and its info.
Sounds simple enough.