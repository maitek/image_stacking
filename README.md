# Automatic Image Stacking in OpenCV

Image stacking is a technique in digital photograpy where multiple photos from the same scene are "stacked" together
in order to reduce image noise caused by a high ISO values. This allows for creating a low noise image at low light conditions. In order to fuse multiple images together the pixels in all photos needs to aligned with each other,
which is usually not possible when taking images without a tripod. This python script shows an example of stack multiple images together by first automatically alligning and then calculating a the fused images as a pixel-wise mean of the alligned images. 

In order to allign the images the perspective transforms between the first images and every other images are estimated. 
The perspective transform can be estimated either using [ORB Keypoint matching](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html) (faster) or
[ECC maximization](http://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#findtransformecc) (more precise).

![alt text](https://github.com/maitek/image_stacking/blob/master/match.jpg "Match keypoints")

## Requirements
- Python 2 or 3
- OpenCV 3.0

## Example Usage:
### ORB method (faster):

```
  python auto_stack.py images/ result.jpg --method ORB
```
![alt text](https://github.com/maitek/image_stacking/blob/master/result_orb.jpg "ORB result image")


### ECC method (more precise):
```
  python auto_stack.py images/ result.jpg --method ECC
```
![alt text](https://github.com/maitek/image_stacking/blob/master/result_ECC.jpg "ECC result image")


