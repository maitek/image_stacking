import argparse
import os
import cv2
import numpy as np
from time import time
from skimage.exposure import is_low_contrast


def loadImages(path, filter_contrast=False):
    file_list = os.listdir(path)
    orig_file_list = [os.path.join(path, x)
                 for x in file_list if x.endswith(('.jpg', '.png', '.bmp', '.tiff'))]

    if filter_contrast:
        file_list = [
            x for x in orig_file_list if not is_low_contrast(cv2.imread(x))]

        orig_file_list.sort()
        file_list.sort()

        exclusion = list(set(orig_file_list) - set(file_list))

        if file_list == orig_file_list:
            print(f"All images good using all of them")
        elif len(file_list) < len(orig_file_list) and len(file_list) > 0:
            print(f"Excluding {exclusion}")
        else:
            print(f"Everything is low contrast, attempting to use all images but 0")
            file_list = orig_file_list
            file_list.remove(f"{path}/0.dng.tiff")
    else:
        file_list = orig_file_list

    return file_list


# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECC(file_list):
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None

    for file in file_list:
        image = cv2.imread(file, 1).astype(np.float32) / 255
        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            stacked_image = image
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            stacked_image += image

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(file_list):

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        print(file)
        image = cv2.imread(file, 1)
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


# ===== MAIN =====
# Read all files in directory


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument(
        '--method', help='Stacking method ORB (faster) or ECC (more precise)')
    parser.add_argument('--show', help='Show result image',
                        action='store_true')
    parser.add_argument('--filter_contrast',
                        help='Filter low contrast images', action='store_true')
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print(f"ERROR {image_folder} not found!")
        exit()
    
    if image_folder.endswith('/'):
        image_folder = image_folder[:-1]

    file_list = loadImages(image_folder, args.filter_contrast)

    if args.method is not None:
        method = str(args.method)
    else:
        method = 'ORB'

    tic = time()

    if method == 'ECC':
        # Stack images using ECC method
        description = "Stacking images using ECC method"
        print(description)
        stacked_image = stackImagesECC(file_list)

    elif method == 'ORB':
        # Stack images using ORB keypoint method
        description = "Stacking images using ORB method"
        print(description)
        stacked_image = stackImagesKeypointMatching(file_list)

    else:
        print(f"ERROR: method {method} not found!")
        exit()

    print(f"Stacked {len(file_list)} images in {time()-tic} seconds")
    print(file_list)

    print(f"Saved {args.output_image}")
    cv2.imwrite(str(args.output_image), stacked_image)

    # Show image
    if args.show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
