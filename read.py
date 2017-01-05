import cv2
import os
import sys
import shutil
import operator
import numpy as np
import re
import glob
from skimage.measure import structural_similarity as ssim
from skimage import color;
import matplotlib.pyplot as plt
#from skimage.measure import compare_ssim as ssim
from joblib import Parallel, delayed
import multiprocessing
import logging
import math

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def feature_compare(thr_id, dir, file_cnt, window, isSlide):
    MIN_MATCH_COUNT = 100
    MAX_UNMATCH_COUNT = 100
    if isSlide:
	offset = int(window/2); 
	start = min(thr_id*window-offset, file_cnt); 
	if start<0:
		start=0; 	
	end = min(start+window,file_cnt);
    else:
	start=min(thr_id*window+1, file_cnt);
    	end=min(start+window,file_cnt);
    scores =[];
   # print start
    exit=0;
    retain =[];	
    if start>=end:
	return;
    img1 = cv2.imread(os.path.join(dir, '%d.png') % start)
    for i in range(start+1,end):
    	img2 = cv2.imread(os.path.join(dir, '%d.png') % i)
	score = feature_matching(img1,img2)
	if i==start+1 and score > 800:
		exit=1;
		select=i-start-1;		
		break;        
	scores.append(score)
        img1= img2
    #    print "tick \n"
    if len(scores) == 0:
	return 
    if len(scores) == 1:
	  return os.path.join(dir, '%d.png') % (start)
    if exit!=1:
	   # print scores
	    min_index, min_score = min(enumerate(scores), key=operator.itemgetter(1))
	    if min_score >= MIN_MATCH_COUNT:
		#best case
		max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))
		# find image number based on index
		#if len(scores)==1:
		#	select=0;
		if max_index==0:
			select=0;
		elif max_index==window-2:
			select=window-1;
		elif scores[max_index-1]< scores[max_index+1]:
			select = max_index+1;
		else: 
			select = max_index; 
	    else:
		# easy case
		#if len(scores)==1:
		#	select=0;
		#print scores
		if min_index==0:
			if scores[1] < MAX_UNMATCH_COUNT:
				select=1;
			else:
				select=0;
		elif min_index==window-2:
			if scores[min_index-1] < MAX_UNMATCH_COUNT:
				select=window-2;
			else:
				select=window-1;
		elif scores[min_index-1]< scores[min_index+1]:
			select= min_index;
		else:
			select= min_index+1;
		# find image number based on index		
    #shutil.copyfile(os.path.join(dir, '%d.png') % (start+select), os.path.join(dir+'/updated/', '%d.png') % (start+select))	
    return os.path.join(dir, '%d.png') % (start+select)


def feature_matching(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create() 
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
    	if m.distance < 0.7*n.distance:
        	good.append(m)

    return len(good)

    #if len(good)>MIN_MATCH_COUNT:
	    #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    #dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	    #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	    #matchesMask = mask.ravel().tolist()

	    #h,w,t  = img1.shape
	    #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    #dst = cv2.perspectiveTransform(pts,M)

	    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	#    return 1	

    #else:
	    #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	    #matchesMask = None
    #	    return 0	

    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
     #              singlePointColor = None,
      #             matchesMask = matchesMask, # draw only inliers
       #            flags = 2)

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    # plt.imshow(img3, 'gray'),plt.show()

	
def feature_pass(dir, window):
    path = dir
    logging.info("----------Feature matching pass started-------")	
    num_files = len([f for f in os.listdir(path)
    	if os.path.isfile(os.path.join(path, f))])
    num_of_iterations= num_files/window
    num_of_processors= multiprocessing.cpu_count() -1;	
    logging.info("parallelization count " + str(num_of_processors))
    logging.info("No of iterations " + str(num_of_iterations))
    selected = []	
    for i in range(0,num_of_iterations, num_of_processors):
		selected = selected + Parallel(n_jobs=num_of_processors)(delayed(feature_compare)(j,dir,num_files, window, False) for j in range(i,i+num_of_processors))

    remaining = num_files % window	
    if remaining != 0 :
    	selected = selected + Parallel(n_jobs=remaining)(delayed(feature_compare)(j,dir,num_files, window, False) for j in range(i,i+remaining))
	
    logging.info("----------Sliding window for Feature matching started-------")	
    for i in range(1,num_of_iterations, num_of_processors):
		selected = selected + Parallel(n_jobs=num_of_processors)(delayed(feature_compare)(j,dir,num_files, window, True) for j in range(i,i+num_of_processors))


    logging.info("----------Feature matching pass finished-------")	

    return selected



def strctural_similarity(thr_id,window_size,sorted_files, isSlide):
	if isSlide:
		offset = int(window_size/2); 
		start = min(thr_id*window_size-offset, len(sorted_files)-1); 
		if start<0:
			start=0; 
		end = min(start + window_size , len(sorted_files))
    	else:	
		start = min(thr_id*window_size, len(sorted_files)-1)
		end = min(start + window_size , len(sorted_files))
	if start>=end:
		return	
	img1 = cv2.imread(os.path.join(sorted_files[start]))
	img1 = color.rgb2gray(img1)
	scores = []	
	for i in range(start+1,end):
		#print "Current File Being Processed is: " + sorted_files[i]
       		img2 = cv2.imread(os.path.join(sorted_files[i]))
		img2 = color.rgb2gray(img2)		
		s = ssim(img1,img2)
		scores.append(s)
		img1=img2;
	#FIXME: Remove me
	if len(scores)==0: 
		return 
	if len(scores)==1: 
		return os.path.join(sorted_files[start])	
	max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))
	# find image number based on index
	if max_index==0:
		if round(scores[max_index],1) == round(scores[max_index+1],1) :
			select=1;
		else :
			select=0;
	elif max_index==len(scores)-1:
		if round(scores[max_index],1) == round(scores[max_index-1],1) :
			select=max_index;
		else: 
			select=max_index+1;
	else: 
		if round(scores[max_index],1) == round(scores[max_index-1],1) :
			select=max_index;
		elif round(scores[max_index],1) == round(scores[max_index+1],1) :
			select=max_index+1;
		else:
			select=-1
	if select!=-1:
		#temp = []
		#num = [] 
		#num = re.findall(r'\d+',str(sorted_files[start+select]))
		#for i in range(start,end):
		#	if i!= start+select:
		#		os.remove(os.path.join(sorted_files[i]))
		#shutil.copyfile(os.path.join(sorted_files[start+select]), os.path.join(dir+'updated_ss/'+str(num[0])+'.png'))	
		#print "selected " + str(sorted_files[start+select])
		return os.path.join(sorted_files[start+select])			

def ss_pass(selected_files, window):
        #print "Image files are: "
	logging.info("----------SSIM pass started-------")	
	sorted_files = sorted(selected_files, key = lambda x: x.split('/')[-1])
	num_of_iterations = len(sorted_files) / window
	num_of_processors= multiprocessing.cpu_count()-1;	
	selected = []
    	logging.info("parallelization count " + str(num_of_processors))
    	logging.info("No of iterations " + str(num_of_iterations))
    	for i in range(0,num_of_iterations, num_of_processors):
	    	selected = selected + Parallel(n_jobs=num_of_processors)(delayed(strctural_similarity)(j,window,sorted_files, False) for j in range(i,i+num_of_processors))	
	remaining = len(sorted_files) % window	
	
	if remaining != 0 :
    		selected = selected + Parallel(n_jobs=remaining)(delayed(strctural_similarity)(j,window,sorted_files, False)for j in range(num_of_iterations,num_of_iterations+remaining))

	logging.info("----------Sliding window for SSIM started-------")	
	for i in range(1,num_of_iterations, num_of_processors):
	    	selected = selected + Parallel(n_jobs=num_of_processors)(delayed(strctural_similarity)(j,window,sorted_files, True) for j in range(i,i+num_of_processors))


	logging.info("----------SSIM pass finished-------")	

	#print "selected "
	#print selected
	return selected 


def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    logging.info("----------Video reading started-------")	
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 1			
    if success:
	    prev_image=image;	
	    while vidcap.isOpened():
		success, image = vidcap.read()
		if not success:
			break;
		err = mse(prev_image,image)
		if err>3500:
			#skip = feature_matching(prev_image, image)	
			#img1 = color.rgb2gray(prev_image)
			#img2 = color.rgb2gray(image)
			#sim = ssim(img1, img2)	
			#if skip==0:	
			cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
			count += 1
			prev_image= image
			#else:
			#	print "\n skipping image due to features"
			#    prev_image= image		
		#else:
			#    print "\n skipping image"
        	
    cv2.destroyAllWindows()
    vidcap.release()
    logging.info("----------Video reading complete-------")	
    return count-1	
'''
Intuition

(N/a + (N/a) -1 )/b +  (N/a + (N/a) -1 )/b -1

2(N/a + (N/a) -1 )/b -1 ~ 20

2(N/a + (N/a) -1 )/b ~ 21

(N/a + (N/a) -1 )/b ~ 21/2

N/a + (N/a) -1 ~ 21/2*b

2N/a ~ 21/2*b +1

2N/a ~ 21/2*b + 1

b = 2a

2N/a ~ 21*a + 1

2N/a ~ 21*a +1

2N/a ~ 21*a 

2N ~ 21*a*a

2N/21 ~ a**2

a =sqrt(2N/21)


'''
def compute_window_sizes(frames_cnt):
	logging.info("Frames selected from video: "+ str(frames_cnt)) 
	if(frames_cnt >= 20):
		#temp = frames_cnt / 10;
		a =  math.floor(math.sqrt(2*frames_cnt/21));
		b = 2*a;		
		a = int(a)
		b = int(b)
		if(a==2):
			a=3;
			b=3;
		if(a==1):
			b=3;
	else :
		a=-1
		b=-1
	logging.info("Window_1 size: "+ str(a)) 	
	logging.info("Window_2 size: "+ str(b)) 		
	return a, b;	

def start_extraction(video, dir):
	#FIXME: Uncomment me once testing is done.	
	name = video.split('/')[-1]
	frames_cnt = video_to_frames(video, dir);
	logging.info("Frames selected from video: "+ str(frames_cnt)) 	
	window_1, window_2 = compute_window_sizes(frames_cnt);
	if(window_1 > 1):	
		selected = feature_pass(dir,window_1)
	else:
		selected = []
		for filename in os.listdir(dir):
   		  selected.append(os.path.join(dir,filename))
	selected = filter(None, selected)
	logging.info("Files selected after feature detection "+str(len(selected)))
	logging.info(selected)	
	if(window_2 > 1):	
		selected_further = ss_pass(selected,window_2)
	else:
		selected_further = selected
	selected_further = filter(None, selected_further)
	logging.info("Files selected after ssim"+str(len(selected_further)))
	logging.info(selected_further)	
	return selected_further
