# import re
# f = open('conv.txt', 'r')
# fo = open('output.txt', "rw+")
# fo.truncate()
# for l in f:
# 	nl = re.split(r'\t+', l.rstrip('\t'))
# 	for res in nl:
# 		res = res.replace("\n", "")
# 		fo.write(res+'\n')
# 		
# 
import sys  
from spacy.en import English
import json
import re

reload(sys)  
sys.setdefaultencoding('utf8')





f = open('final.txt', 'r')
fo = open('foutput.txt', "rw+")
fchanged = open('changed.txt', "rw+")
parser = English(parser= False)
conds = ["PERSON", "GPE"]
for l in f:
	r = re.sub(r'[0-9]+', 'NUMBER', l)
	r = unicode(r,encoding="utf-8")
	rparsed = parser(r)
	ents = list(rparsed.ents)
	for entity in ents:
		# print(entity.label, entity.label_, ' '.join(t.orth_ for t in entity))
	    # print( entity.label_, ' '.join(t.orth_ for t in entity))
	    xy =  ' '.join(t.orth_ for t in entity)
	    # print xy
	    fchanged.write((entity.label_)+ ' '.join(t.orth_ for t in entity)+'\n')
	    if(entity.label_ in conds):
	    	if(entity.label_ == "GPE"):
	    		r = r.replace(xy, 'PLACE')
	    	else:
	    		r = r.replace(xy, entity.label_)
	fo.write(r)