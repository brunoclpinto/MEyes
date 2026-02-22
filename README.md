# MEyes
Mobile eyes, an app to support visually impaired by processing what they can't see through the eyes of their wearables and mobile devices.

## Warnings
This tool is to assist and try to make your life easier and more independent, it is by no means foolproof and its help should be taken as suggestion and not a certainty.  
Be cautions and keep yourself safe.

## Confidence Level
In front of every feature and their intermediate steps, an emoji will show how confident i am of its capacity and accuracy, check what they mean.

🤩 - Freakin miracle worker. No fails.  
😁 - ITSSS GREEAAATTT!!. Works as expected with minor fails and/or false positives.  
🫩 - Kinda works. Works close to expected with limitations, some fails and/or false positives.  
😦 - Spotty. Yeah, works some but not fully or consistently.  
😱 - HELL NO!!!. Mostly unusable and inconsistent.

## Where will it work
Well, since models can be converted and logic can be translated into different languages, lets go and say everywhere.

## What can it do?

### Bus 🫩
Yep, detects if bus is coming and tries to process its number/destination information.

#### Technical docs

#### Bus is comming!!! 😁
Detects bus is on route towards you, combined with simple bounding box size changes for tracking direction.  
Each detected bus throughout a session will receive an iterated ID starting at Bus1, to try and separate whats been detected.  

Expect some false positives, YOLO detects Tram, large trucks, campers, delivery trucks and trolleys as Bus.

#### Read the info 😦
It does detect and performs OCR, but results are quite spotty, wrong for most of the time and only correct and clear when its way too close.  
To improve accuracy and reduce amount of bad information it currently only speaks out the bus number.

## What am working on?

## What will it do?
### Crosswalk Assistance 
### Read buttons and device screens
Goal is to be able to read button options on a device, elevator or whatever.
Should be able to read all options and single my choice by pointing finger.
Believe this one is gonna be fun to make it work, we'll see.

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
Convert segmentation into 2D viable path matrix where fastest path can be easily calculated.
Maybe i'll make it work, would be awesome though!!