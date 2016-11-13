from read import start_extraction
from revsr import start_search
import logging
import os
import shutil

name = 'modi.mp4'
video = name
path = ''
dir = './extract_modi/'
print "\n\n ---------------- ViTag --------------"
logging.basicConfig(filename='ViTagLog_'+name+'.log',level=logging.DEBUG)
logging.info("\n\n ---------------- ViTag --------------")
logging.info("\n\n ---------------- Logging information--------------")
logging.info("\n\n -----Video: "+name+" path: "+path)
if os.path.exists(dir):
	shutil.rmtree(dir);
os.makedirs(dir)
files = start_extraction(video,dir)
results = start_search(files)
output = list(set(results)) 
print output
print "\n\n \t\t Developed At IIT Hyderabad, India"
print "\n\n \t\t Abhishek Patwardhan, Santanu Das, Sakshi Varshney, Maundendra Desarkar"
print "\n\n \t\t For Bugs contact : cs15mtech11015@iith.ac.in"
logging.info("Results are ")
logging.info(results)
logging.info("Output shown")
logging.info(output)
logging.info("Program executed successfully")
logging.info("\n\n ----------------END--------------")
logging.info("\n\n ----------------ViTag--------------")
logging.info("\n\n \t\t Developed At IIT Hyderabad, India")
logging.info("\n\n \t\t Abhishek Patwardhan, Santanu Das, Sakshi Varshney, Maundendra Desarkar")
logging.info("\n\n \t\t For Bugs contact : cs15mtech11015@iith.ac.in")

