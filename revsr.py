#
#	Developed at IIT Hyderabad
#	Abhishek, Santanu, Sakshi, Dr. Maunendra
#
#
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
from joblib import Parallel, delayed
import multiprocessing

def retrieve(image_url):
    returned_code = StringIO()
    conn = pycurl.Curl()
    conn.setopt(conn.URL, str(image_url))
    conn.setopt(conn.FOLLOWLOCATION, 1)
    conn.setopt(conn.USERAGENT, 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.97 Safari/537.11')
    conn.setopt(conn.WRITEFUNCTION, returned_code.write)
    conn.perform()
    conn.close()
    textHtml = returned_code.getvalue()
    return google_image_results_parser(textHtml)


# Parses returned code (html,js,css) and assigns to array using beautifulsoup
def google_image_results_parser(code):
    soup = BeautifulSoup(code)
    whole_array = {
                   'original' :[],
                   'description':[],
                   'title':[],
                   'result_qty':[]}
    for orig in soup.findAll('a', attrs={'style':'font-style:italic'}):
        whole_array['original'].append(orig.get_text())


    
    gtext = ' '.join(whole_array['original'])
    logging.info(" Google text : " + str(gtext))
    return str(gtext);

def processInput(path):
    filePath = path
    searchUrl = 'http://www.google.com/searchbyimage/upload'
    multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
    response = requests.post(searchUrl, files=multipart, allow_redirects=False)
    fetchUrl = response.headers['Location']
    return retrieve(fetchUrl)

def start_search(files):
	path = dir
	num_files = len(files)
	logging.info("---------- No. of files: "+str(num_files))
        inputs = range(num_files)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(path) for path in files)
	return results
