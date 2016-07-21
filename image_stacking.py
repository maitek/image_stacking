import os
import cv2
import numpy as np
from time import time

image_folder = "images/"

#
file_list = os.listdir(image_folder)
file_list = [os.path.join(image_folder, x) for x in file_list if x.endswith(".jpg")]

fast = cv2.FastFeatureDetector_create()
orb = cv2.ORB_create()

show_debug = True

# disable OpenCL to fix buggy ORB in OpenCV 3.1
cv2.ocl.setUseOpenCL(False)

avg_image = None
first_kp = None
first_des = None
# find keypoints and descriptors in each image
tic = time()
for file in file_list:
    image = cv2.imread(file, 1)
    imageF = image.astype(np.float32) / 255

    # compute the descriptors with ORB
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    # create BFMatcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if avg_image is None:
        # Save keypoints for first image
        avg_image = imageF
        first_image = image
        first_kp = kp
        first_des = des
    else:
        # Find matches and sort them in the order of their distance
        matches = matcher.match(first_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # draw matches for debugging
        image_matches = image.copy()
        image_matches = cv2.drawMatches(first_image, first_kp, image, kp, matches, image_matches, flags=2)

        src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp [m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate perspective transformation
        M, mask = cv2.findHomography(dst_pts,src_pts,cv2.RANSAC,5.0)
        
        # Compensate image for movements
        w, h, _ = imageF.shape

        corrected_image = cv2.warpPerspective(imageF ,M, (h,w))
        
        avg_image += corrected_image

        if show_debug:
            cv2.imshow("corrected", corrected_image)
            #cv2.imshow("image_matches", image_matches)
            cv2.waitKey(0)

    image = cv2.drawKeypoints(image, kp, image, color=(255, 0, 0))

avg_image /= len(file_list)

print("Stacking " + str(len(file_list)) + " images in " + str(time()-tic) + " seconds")
cv2.imshow("Stacked image", avg_image)
cv2.waitKey(0)
cv2.imwrite("result.jpg",(avg_image*255).astype(np.uint8))
