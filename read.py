#
#	Developed at IIT Hyderabad
#	Abhishek, Santanu, Sakshi, Dr.Maunendra
#
#
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
from joblib import Parallel, delayed
import multiprocessing
import logging
import math

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


def feature_compare(thr_id, dir, file_cnt, window):
    MIN_MATCH_COUNT = 100
    MAX_UNMATCH_COUNT = 100
    start=min(thr_id*window+1, file_cnt);
    end=min(start+window,file_cnt);
    scores =[];
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
    if len(scores) == 0:
	return 
    if len(scores) == 1:
	  return os.path.join(dir, '%d.png') % (start)
    if exit!=1:
	    min_index, min_score = min(enumerate(scores), key=operator.itemgetter(1))
	    if min_score >= MIN_MATCH_COUNT:
		max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))
		if max_index==0:
			select=0;
		elif max_index==window-2:
			select=window-1;
		elif scores[max_index-1]< scores[max_index+1]:
			select = max_index+1;
		else: 
			select = max_index; 
	    else:
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
    return os.path.join(dir, '%d.png') % (start+select)


def feature_matching(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create() 
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
    	if m.distance < 0.7*n.distance:
        	good.append(m)

    return len(good)
	
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
		selected = selected + Parallel(n_jobs=num_of_processors)(delayed(feature_compare)(j,dir,num_files, window) for j in range(i,i+num_of_processors))

    remaining = num_files % window	
    if remaining != 0 :
    	selected = selected + Parallel(n_jobs=remaining)(delayed(feature_compare)(j,dir,num_files, window) for j in range(i,i+remaining))
	
    logging.info("----------Feature matching pass finished-------")	
    return selected




def strctural_similarity(thr_id,window_size,sorted_files):
	start = min(thr_id*window_size, len(sorted_files)-1)
	end = min(start + window_size , len(sorted_files))
	if start>=end:
		return	
	img1 = cv2.imread(os.path.join(sorted_files[start]))
	img1 = color.rgb2gray(img1)
	scores = []	
	for i in range(start+1,end):
       		img2 = cv2.imread(os.path.join(sorted_files[i]))
		img2 = color.rgb2gray(img2)		
		s = ssim(img1,img2)
		scores.append(s)
		img1=img2;
	if len(scores)==0: 
		return 
	if len(scores)==1: 
		return os.path.join(sorted_files[start])	
	max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))
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
		return os.path.join(sorted_files[start+select])			

def ss_pass(selected_files, window):
	logging.info("----------SSIM pass started-------")	
	sorted_files = sorted(selected_files, key = lambda x: x.split('/')[-1])
	num_of_iterations = len(sorted_files) / window
	num_of_processors= multiprocessing.cpu_count()-1;	
	selected = []
    	logging.info("parallelization count " + str(num_of_processors))
    	logging.info("No of iterations " + str(num_of_iterations))
    	for i in range(0,num_of_iterations, num_of_processors):
	    	selected = selected + Parallel(n_jobs=num_of_processors)(delayed(strctural_similarity)(j,window,sorted_files) for j in range(i,i+num_of_processors))	
	remaining = len(sorted_files) % window	
	
	if remaining != 0 :
    		selected = selected + Parallel(n_jobs=remaining)(delayed(strctural_similarity)(j,window,sorted_files)for j in range(num_of_iterations,num_of_iterations+remaining))

	logging.info("----------SSIM pass finished-------")	
	return selected 


def video_to_frames(video, path_output_dir):
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
			cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
			count += 1
			prev_image= image
    cv2.destroyAllWindows()
    vidcap.release()
    logging.info("----------Video reading complete-------")	
    return count-1	

def compute_window_sizes(frames_cnt):
	logging.info("Frames selected from video: "+ str(frames_cnt)) 
	if(frames_cnt > 500):
		temp = frames_cnt / 10;
		a =  math.floor(math.sqrt(temp/2));
		b = 2*a;
	else :
		temp = frames_cnt / 10;
		a =  math.floor(math.sqrt(temp/2));
		b = 2*a;
	if(a<=2):
		a=3
		b=5
	a = int(a)
	b = int(b)
	logging.info("Window_1 size: "+ str(a)) 	
	logging.info("Window_2 size: "+ str(b)) 		
	return a, b;	

def start_extraction(video, dir):
	name = video.split('/')[-1]
	frames_cnt = video_to_frames(video, dir);
	logging.info("Frames selected from video: "+ str(frames_cnt)) 	
	window_1, window_2 = compute_window_sizes(frames_cnt);
	selected = feature_pass(dir,window_1)
	selected = filter(None, selected)
	logging.info("Files selected after feature detection "+str(len(selected)))
	logging.info(selected)	
	selected_further = ss_pass(selected,window_2)
	selected_further = filter(None, selected_further)
	logging.info("Files selected after ssim"+str(len(selected_further)))
	logging.info(selected_further)	
	return selected_further
