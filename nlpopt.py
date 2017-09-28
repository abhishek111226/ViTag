import requests
from requests.auth import HTTPDigestAuth
import json
import nltk
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import heapq
import math
import logging
import sys

def callMicrosoftAPI(keyword, topk):
    url = 'https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance='+keyword+'&topK='+str(topk)+'&api_key=5XCchPffKolWBm3XnctawsmTDpgGGRTw'
    print 'Calling Microsoft API for '+keyword
    #print url
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        jasonData = json.loads(response.content)
        print("The response contains {0} properties".format(len(jasonData)))
        
        if(topk==5):
		conceptwords = dict()
		for key, value in jasonData.items():
		    #if(round(value,2) >= 0.1):	
		    conceptwords[key.encode("ascii")] = round(value,2)    
		    #conceptwords.append(key.encode("ascii"))
	elif(topk==15):
		jasonData = json.loads(response.content)
		#print("The response contains {0} properties".format(len(jasonData)))
		conceptwords = []
		for key, value in jasonData.items():	
                	conceptwords.append(key) 
        else:
        	print "Unhandled case"
        	sys.exit(0)
        	  
    else:
        print 'error response from API for '+keyword
    return conceptwords

def find_semantic_similarity(word1, word2):
    list1 = callMicrosoftAPI(word1, 15)
    list2 = callMicrosoftAPI(word2, 15)
    print set(list1).intersection(list2)
    return len(set(list1).intersection(list2))    

def process_graph(rows, columns, matrix):
    cost = [0] * len(rows)
    print matrix
    for r in rows:	
	    for c in columns: 
	    	cost[rows.index(r)] += matrix[rows.index(r)][columns.index(c)] 		

    print "Initial cost"
    print cost
    #return cost
    #print "Finding similarity matrix"
    semant_sim = [[0 for x in range(len(columns))] for y in range(len(columns))] 
    factor=0;	
    for r in range(len(columns)):
	for c in range(len(columns)):
	        if r!= c: 
			semant_sim[r][c] = semant_sim[c][r] = find_semantic_similarity(columns[r], columns[c]) 
			factor = semant_sim[r][c] if (factor < semant_sim[r][c]) else factor  	
		else:
			semant_sim[r][c] = semant_sim[c][r] = 1
    #print "similarity matrix"	
    #print semant_sim		
    if(factor!=0):
	    for r in range(len(columns)):
		for c in range(len(columns)):
			if r!= c: 
	    			semant_sim[r][c] = semant_sim[r][c]/ (factor * 1.0)
    #print "Normalized: "					
    #print semant_sim
    for r in range(len(columns)):
	for c in range(len(columns)):
	        if r!= c and semant_sim[r][c] > 0:
	        	for k in range(len(rows)):
	        		cost[k] += semant_sim[r][c]* matrix[k][c] 
	        	 
    #print "Final cost"
    #print rows
    #print cost	        	
    return cost	

def nlp_opt(output):
    mainwords = output
    #print 'Main words:'
    print mainwords
    multiple_wordstring = []
    tag_to_concept = {};	
    for word in mainwords:
	print word
	print len(word.split())
        if len(word.split()) == 1:
	    tag_to_concept[word] = callMicrosoftAPI(word, 5) 	
            #conceptwords = conceptwords + callMicrosoftAPI(word)    
        else:
            #save queries with multiple words in a separate list 
            multiple_wordstring.append(word)
            
    stops = set(stopwords.words('english'))      
    #check if the single words are a substring of the multiple_words
    for long_word in multiple_wordstring:
	conceptwords = callMicrosoftAPI(long_word, 5)   
	factor = len(long_word.split())     
	for term in long_word.split(' '):
		if term not in stops and tag_to_concept.has_key(term):
			temp = {}
			for key in tag_to_concept.get(term):
				temp[key] = tag_to_concept.get(term).get(key, 0) / factor
			conceptwords = dict(temp.items() + conceptwords.items() +[(k, temp[k] + conceptwords[k]) for k in set(conceptwords) & set(temp)])    
	tag_to_concept[long_word]= conceptwords             

    logging.info("\n\n -----dictionary created for nlpOpt: ")
    logging.info(tag_to_concept)
    print "-----dictionary created for nlpOpt: "
    print tag_to_concept
    print "-----dictionary created for nlpOpt: "
    rows = []	
    for key in tag_to_concept:
	rows = rows + list(set(tag_to_concept.get(key).keys()))	
	rows = list(set(rows))

    if(len(rows)==0):
    	logging.info("Unable to perform nlpOpt simply because Concepts cannot be found")
    	return output 

    columns = list(tag_to_concept.keys())

    w, h = len(columns), len(rows);
    matrix = [[0 for x in range(w)] for y in range(h)] 	
    for r in rows:
	for c in columns:	
		if( r in tag_to_concept.get(c).keys()):
			matrix[rows.index(r)][columns.index(c)]= tag_to_concept.get(c).get(r);
		else:
			matrix[rows.index(r)][columns.index(c)] = 0
    logging.info("\n\n [nlpOpt] Rows are:  ")
    logging.info(rows)
    logging.info("\n\n [nlpOpt] Columns are:  ")
    logging.info(columns)
    logging.info("\n\n [nlpOpt] Matrix :  ")
    logging.info(matrix)
    
    print rows
    print matrix

    cost = process_graph(rows, columns, matrix)
    
    print cost 
    print rows
    temp = max(cost);
    for k in range(0,len(cost)):
    	if cost[k] == temp:
    		print rows[k]
    		
    selected = []
    k = int(math.ceil((len(output) * 0.33333)))
    print "k="+str(k)
    logging.info("\n\n [nlpOpt] topK :  ")
    logging.info(k)

    topk = heapq.nlargest(k, cost) 
    logging.info("\n\n [nlpOpt] topK selected :  ")
    logging.info(topk)
    print "topk ="+str(topk)
    threshold = sum(cost) ** (1/len(cost)) /4.0   
    logging.info("\n\n [nlpOpt] threshold selected :  ")
    logging.info(threshold)
    print "threshold="+str(threshold)
    for k in topk:
            if(k >= threshold):    		
	    	selected.append(rows[cost.index(k)])
    logging.info("\n\n [nlpOpt] Returning from NlpOpt : ")
    logging.info(output + selected)
    return output+selected	
