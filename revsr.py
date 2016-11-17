import pycurl, json
from flask import Flask, url_for, json, request
from StringIO import StringIO
from bs4 import BeautifulSoup
import requests
from os import listdir
from os.path import isfile, join
import os.path
import sys
import logging
#for parallelizing
from joblib import Parallel, delayed
import multiprocessing

# retrieves the reverse search html for processing. This actually does the reverse image lookup
def retrieve(image_url):
    returned_code = StringIO()
    #full_url = "https://www.google.com/searchbyimage?&image_url=" + image_url
    #print "Accessing Url"
    #print image_url
    #print "\n"
    try:
        conn = pycurl.Curl()
        conn.setopt(conn.URL, str(image_url))
        conn.setopt(conn.FOLLOWLOCATION, 1)
        conn.setopt(conn.USERAGENT, 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.97 Safari/537.11')
        conn.setopt(conn.WRITEFUNCTION, returned_code.write)
        conn.perform()
        conn.close()
    except Exception, e:
        logging.info( "Failed to get info from Google: "+str(e))
        google_image_results_parser("")
        
    textHtml = returned_code.getvalue()
    return google_image_results_parser(textHtml)


# Parses returned code (html,js,css) and assigns to array using beautifulsoup
def google_image_results_parser(code):
    try:
        soup = BeautifulSoup(code)
        # initialize 2d array
        whole_array = {
                   'original' :[],
                   'description':[],
                   'title':[],
                   'result_qty':[]}
        #actual search text
        for orig in soup.findAll('a', attrs={'style':'font-style:italic'}):
            whole_array['original'].append(orig.get_text())
    
        gtext = ' '.join(whole_array['original'])
        logging.info(" Google text : " + str(gtext))
    
    except Exception, e:
        logging.info("Failed to parse google response: "+str(e))
        return ""
    return str(gtext);

def processInput(path, num_files):
    try:

        filePath = path
        #print "REVERSE IMAGE SEARCH For "+path
        searchUrl = 'http://www.google.com/searchbyimage/upload'
        multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
        response = requests.post(searchUrl, files=multipart, allow_redirects=False)
        fetchUrl = response.headers['Location']

    except Exception, e:
        logging.info("Failed to encode image search URL:" +str(e))
	return ""

    return retrieve(fetchUrl)
    

def start_search(files):	
	path = dir
	num_files = len(files)
	logging.info("---------- No. of files: "+str(num_files))
        
        inputs = range(num_files)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(path, num_files) for path in files)
        
        if len(results)==0:
            logging.info("****No result from any image search****")
	
        if len(results) < num_files:
            logging.info("Search successful for "+str(len(results))+" out of "+str(num_files)+" images....")
	return results
	   
