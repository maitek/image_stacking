# Automatic Image Stacking in OpenCV

Image stacking is a technique in digital photograpy where multiple from the same scene are "stacked" together
in order to reduce image noise caused by a high ISO values. This allows for creating low noise low at low light conditions.
In order to fuse multiple images together the pixels in all photos needs to aligned with each other,
which is usually not possible when taking images without a tripod.

This python script shows an example of stack multiple images together by first automatically alligning and then calculating
a the fused images as a pixel-wise mean of the alligned images. 

In order to allign the images the perspective transforms between the first images and every other images is estimated. 
The perspective transform can be estimated either using ORB Keypoint matching (faster) or
Enhanced Correlation Coefficient Maximization (more precise).

## Requirements
- Python 2 or 3
- OpenCV 3.0

## Example Usage

```
  python image_stacking.py image_folder/ result.jpg --method ECC
```

