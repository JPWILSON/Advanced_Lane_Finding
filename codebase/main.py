import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import glob



# Add all of the methods required 
# First, fix the distortion
image5 = mpimg.imread('test5.jpg')
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

#	Now that we have the matrices (imgpoints and objpoints) & have calibrated, we can undistort the images
def undistort(img, mtx, dist):
	#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[:2]), None, None)
	undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)
	return undistorted_image


# Now define the methods for the 3 gradient threshold techniques
def abs_sobel_thresh(img, orient='x', s_kernel = 3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = s_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = s_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary 

def mag_thresh(img, s_kernel= 3, thresh=(0, 255)): 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = s_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = s_kernel)
    gradmag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag < thresh[1])] = 1
    return binary_output 

def dir_thresh(img, s_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = s_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = s_kernel))
    grad_dir = np.arctan2(abs_sobelx, abs_sobely)
    bin_mask = np.zeros_like(grad_dir)
    bin_mask[(grad_dir > thresh[0]) & (grad_dir < thresh[1])] = 1
    return bin_mask

# Now define a method for combining the three methods: 
def combine_grad_thresh(img):
	#Run above methods with selected parameters for thresh & ksize
	gradx = abs_sobel_thresh(img, orient='x', s_kernel=3, thresh=(20, 100))
	grady = abs_sobel_thresh(img, orient='y', s_kernel=3, thresh=(5, 100))
	mag_binary = mag_thresh(img, s_kernel=5, thresh=(30, 100))
	dir_binary = dir_thresh(img, s_kernel=11, thresh=(0.7, 1.3))
	#make a combination of them...
	combined_sobel = np.zeros_like(dir_binary)
	combined_sobel[(gradx == 1) & (grady == 1) | ((mag_binary == 1) & ( dir_binary ==1)) ] = 1
	return combined_sobel

# Color thresholding, taking the saturation channel from the HLS color space
def hls_saturation(img, thresh=(90, 220)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s = hls[:,:,2]
	binary_s = np.zeros_like(s)
	binary_s[(s >= thresh[0]) & (s <= thresh[1])] = 1
	return binary_s


# Now, COMBINE color and grad!!!!
def combined(img):
	grd = combine_grad_thresh(img)
	sat = hls_saturation(img)
	comb_bin = np.zeros_like(grd)
	comb_bin[(grd == 1) & (sat == 1)] = 1
	return comb_bin



# May not be necessary, but here is the method for plotting the output
def plotting(img):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=50)
	#NB for now this is just sobel combined, should be col AND sobel combined! FIXED
	combination = combined(img)
	ax2.imshow(combination, cmap='gray')
	ax2.set_title('Thresholded Grad. Dir. Combined with Saturation(hls)s_', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


plotting(image5)


#	The image inputted should now be transformed to a top down view
# 		NB! to keep reviewing the src and dest points!!!
def transform(img):
	(h, w) = (img.shape[0], img.shape[1])
	src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
	dest = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
	M = cv2.getPerspectiveTransform(src, dest)
	trnsformed = cv2.warpPerspective(img, M, (w, h))
	return transformed, M


