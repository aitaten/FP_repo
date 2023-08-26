import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from scipy import ndimage


def clean_radar(image,Thr=0.1,S_area=100):
	"""Function to remove spurious reflectivity (or rain)
    from a rainfall field. There are two tunable parameters:
	INPUT:  image 	= 	rainfall field
			Thr 	=	Threshold for rain (or reflectivity) - 0.1 when not specified by user
			S_area 	=	The precipitation areas smaller than this value are removed.
    OUTPUT: clean_image = rainfall field without spurious rain
    """
	clean_image = image.copy()
	clean_image[clean_image < Thr]=0.

	# apply threshold
	bw = closing(image >= Thr, square(3))

	# remove artifacts connected to image border
	cleared = clear_border(bw)

	# label image regions
	label_image = label(cleared)

	for region in regionprops(label_image):
		# take regions with large enough areas
		if region.area < S_area:
			clean_image[label_image == region.label]=image.min()

	return clean_image

def clean_with_ndimage(image,Thr=0.1,S_area=100.):
	# def clean_radar(image,Thr=0.1,S_area=100.):

	image_cleaned=image.copy()
	image_cleaned[image_cleaned < Thr]=0.

	struct = ndimage.generate_binary_structure(np.ndim(image),2) #2 is used to include diagonal elements
																#1 for not including the diagonal

	labeled_array, num_features = ndimage.label(np.round(image >= Thr).astype('int'), structure=struct)

	if num_features <=1:
		return image

	for i_features in range(1,num_features):
		if np.sum(labeled_array == i_features) < S_area:
			image_cleaned[labeled_array == i_features]=image.min()

	return image_cleaned

def clean_radar_old(image,Thr=0.1,S_area=100):

	x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	indexes = np.dstack([x, y]).reshape(-1,2)

	CS = plt.contour(x,y, image, colors='b', linewidths=2, levels=[Thr])

	clean_image = image.copy()

	for level in CS.collections:
		for kp,path in reversed(list(enumerate(level.get_paths()))):
			# go in reversed order due to deletions!

			# include test for "smallness" of your choice here:
			# I'm using a simple estimation for the diameter based on the
			#    x and y diameter...
			verts = path.vertices # (N,2)-shape array of contour line coordinates
			diameter = np.max(verts.max(axis=0) - verts.min(axis=0))

			if diameter<S_area: # threshold to be refined for your actual dimensions!
				# del(level.get_paths()[kp])  # no remove() for Path objects:(

				within = indexes[path.contains_points(indexes)]
				clean_image[within[:,1],within[:,0]]=image.min()

	return clean_image


if __name__ == "__main__":
	#### EXAMPLE HOW TO RUN AND COMPARE SPEED ####
	# import matplotlib.pyplot as plt
	# from skimage import data
	# image = data.coins()[50:-50, 50:-50]
	# # import f_clean_radar as fcr
	# from f_clean_radar import clean_radar
	# from f_clean_radar import clean_radar_old
	# from clean_radar_data import clean_radar as clean_with_ndimage
	import timeit
	import time
	start = time.time()
	image_clean=clean_radar(image,Thr=100,S_area=50)
	print('Time for clean_radar: ',time.time() - start)
	start = time.time()
	image_clean=clean_with_ndimage(image,Thr=100,S_area=50)
	print('Time for clean_radar: ',time.time() - start)
	start = time.time()
	image_clean_old=clean_radar_old(image,Thr=100,S_area=50)
	print('Time for clean_radar (old): ',time.time() - start)
	def wrapper(func, *args, **kwargs):
	    def wrapped():
	        return func(*args, **kwargs)
	    return wrapped
	wrapped = wrapper(clean_radar, image,Thr=100,S_area=50)
	timeit.timeit(wrapped, number=100)/100.
	wrapped = wrapper(clean_with_ndimage, image,R_Thr=100,min_size=50)
	timeit.timeit(wrapped, number=100)/100.
	wrapped = wrapper(clean_radar_old, image,Thr=100,S_area=50)
	timeit.timeit(wrapped, number=10)/10.
	# timeit.timeit('image_clean=fcr.clean_radar(image,Thr=100,S_area=50)', number=100)
	# timeit.timeit('image_clean_old=fcr.clean_radar_old(image,Thr=100,S_area=50)', number=100)