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
numfiles=0

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

    numfiles += 1
    return str(gtext);


    # Links for all the search results
    #for li in soup.findAll('li', attrs={'class':'g'}):
    #    sLink = li.find('a')
    #    whole_array['links'].append(sLink['href'])

    # Search Result Description
    #for desc in soup.findAll('span', attrs={'class':'st'}):
    #    whole_array['description'].append(desc.get_text())

    # Search Result Title
    #for title in soup.findAll('h3', attrs={'class':'r'}):
    #    whole_array['title'].append(title.get_text())

    # Number of results
    #for result_qty in soup.findAll('div', attrs={'id':'resultStats'}):
    #    whole_array['result_qty'].append(result_qty.get_text())
    

    #return 1 #build_json_return(whole_array)

#def build_json_return(whole_array):
    #return json.dumps(whole_array)

    #print "Google text:"
    #gtext = ' '.join(whole_array['original'])
    #print gtext
    #with open("Output.txt", "a") as myfile:
    #    myfile.write("\n"+gtext)

    #print "\n"
    #print "description:"
    #desc = ' '.join(whole_array['description'])
    #print desc
    
    #print "\n"
    #print "Title:"
    #title = ' '.join(whole_array['title'])
    #print title

    #print "\n"
    #print "results:"
    #print ' '.join(whole_array['result_qty'])

    #print to file
    #text_file = open("Output.txt", "w")
    #text_file.write(title)
    #text_file.close()
    
#if __name__ == '__main__':
#    app.debug = True
#app.run(host='0.0.0.0')

#retrieve("tajmahal.org.uk/gifs/taj-mahal.jpeg")
#retrieve("103.232.241.5/taj.jpeg")

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

    return retrieve(fetchUrl)

def start_search(files):
	path = dir
	num_files = len(files)
	logging.info("---------- No. of files: "+str(num_files))
        
        #delete Output.txt if it exists
        #if os.path.isfile('Output.txt'):
        #    os.remove('Output.txt')

        #get the list of files
        
        inputs = range(num_files)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(path, num_files) for path in files)
        
        if numfiles == 0:
            logging.info("****No result from any image search****")

        if numfiles < num_files:
            logging.info("Search successful for "+numfiles+" out of "+num_files+" images....")

	return results
	#for path in onlyfiles:
	#   processInput(path)    
	   
