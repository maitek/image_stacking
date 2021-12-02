import os
import cv2
import numpy as np
from time import time



# Align and stack images with ECC method
# Slower but more accurate
# Uses mean average to stack
def stackImagesECC(file_list, avg):
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None if avg == 'mean' else []

    for file in file_list:
        if avg == 'mean':
            image = cv2.imread(file,1).astype(np.float32) / 255
        else:
            image = cv2.imread(file,1)

        print(file)
        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if avg == 'mean':
                stacked_image = image
            else:
                stacked_image.append(image)
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            if avg == 'mean':
                stacked_image += image
            else:
                stacked_image.append(image)

    if avg == 'mean':
        stacked_image /= len(file_list)
        stacked_image = (stacked_image*255).astype(np.uint8)
    else:
        stacked_image = np.stack(stacked_image, axis=2)
        stacked_image = np.median(stacked_image, axis=2)

    return stacked_image


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(file_list, avg):

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        print(file)
        image = cv2.imread(file,1)
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image


# Stack images with median averaging; no transform
def stackImagesMedian(file_list):
    first_image = None
    stacked_images = []

    for file in file_list:
        image = cv2.imread(file,1)
        print(file)
        stacked_images.append(image)

    stacked_images = np.stack(stacked_images, axis=2)
    stacked_image = np.median(stacked_images, axis=2)
    return stacked_image


# ===== MAIN =====
# Read all files in directory
import argparse


if __name__ == '__main__':
    avgs = ['mean', 'median']

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument('--method', help='Stacking method ORB (faster), ECC (more precise) or MEDIAN (no transform)')
    parser.add_argument('--avg', help='Averaging method mean or median')
    parser.add_argument('--show', help='Show result image',action='store_true')
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print("ERROR {} not found!".format(image_folder))
        exit()

    file_list = os.listdir(image_folder)
    file_list = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]

    if args.method is not None:
        method = str(args.method)
    else:
        method = 'KP'

    if args.avg is not None:
        avg = str(args.avg)
    else:
        avg = 'mean'

    if avg not in avgs:
        print(f'ERROR: avg {avg} not found!')
        exit()

    tic = time()

    if method == 'ECC':
        # Stack images using ECC method
        description = f'Stacking images using ECC method with {avg} averaging'
        print(description)
        stacked_image = stackImagesECC(file_list, avg)

    elif method == 'ORB':
        #Stack images using ORB keypoint method
        description = f'Stacking images using ORB method with {avg} averaging'
        print(description)
        stacked_image = stackImagesKeypointMatching(file_list, avg)

    elif method == 'MEDIAN':
        #Stack images using median only, no perspective transformation
        description = "Stacking images using MEDIAN method"
        print(description)
        stacked_image = stackImagesMedian(file_list)

    else:
        print("ERROR: method {} not found!".format(method))
        exit()

    print("Stacked {0} in {1} seconds".format(len(file_list), (time()-tic) ))

    print("Saved {}".format(args.output_image))
    cv2.imwrite(str(args.output_image),stacked_image)

    # Show image
    if args.show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
