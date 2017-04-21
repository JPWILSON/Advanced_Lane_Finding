import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import glob


# Add all of the methods required 
# First, fix the distortion

#	Fn for collecting matrix of objp's and image points and calibrating
def get_img_pts_calib(test_image_set):
	objoints = []
	objp = np.eros((6*9, 3), np.float32)
	objp = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
	imgpoints = []
	nx, ny = 9, 6
	for fname in test_image_set:
		img = mpimg.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		if ret == True: 
			objpoints.append(objp)
			imgpoints.append(corners)
			cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[:2]), None, None)
	return nx, ny, mtx, dist 

#	Now that we have the matrices (imgpoints and objpoints) & have calibrated, we can transform images
def undistort(img, mtx, dist):
	#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[:2]), None, None)
	undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)
	return undistorted_image

#The image inputted should 
def transform(img):
	(h, w) = (img.shape[0], img.shape[1])
	src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
	dest = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
	M = cv2.getPerspectiveTransform(src, dest)
	trnsformed = cv2.warpPerspective(img, M, (w, h))
	return transformed, M





