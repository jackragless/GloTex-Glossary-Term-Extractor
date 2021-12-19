import json
all_config = json.load(open("config.json", "r"))
CONFIG = all_config['corpus']
data_loc = all_config['data_loc']

import wikipedia
import wikipediaapi
import pickle
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import concurrent.futures
from text_cleaner import cleanText
import pandas as pd
import pickle
import re
import itertools
import sys
from requests.exceptions import ConnectionError
from wikipedia.exceptions import DisambiguationError, PageError
import nltk
nltk.download('averaged_perceptron_tagger')
import sdow as database
import urllib.parse
if bool(CONFIG['sdow_enabled']):
    import sdow


person_set = set(pd.read_csv(data_loc+'person_ent_list.csv')['name'])
location_set = set(pd.read_csv(data_loc+'location_ent_list.csv')['name'])
ngram_set = set()
ngram_set.update([str(ele) for ele in pd.read_csv(data_loc+'wp_1gram.csv')['string']][:CONFIG['ngram_cutoff']['1']])
ngram_set.update([str(ele) for ele in pd.read_csv(data_loc+'wp_2gram.csv')['string']][:CONFIG['ngram_cutoff']['2']])
ngram_set.update([str(ele) for ele in pd.read_csv(data_loc+'wp_3gram.csv')['string']][:CONFIG['ngram_cutoff']['3']])

wikiminer = wikipediaapi.Wikipedia('en')


###########################################################################################
###########################################################################################
#wiki_mine
###########################################################################################
###########################################################################################


def wikiTitleMiner(title, corpus_size):
	exc_types = ["List of", "Outline of", "Index of"]
	queue = [title]
	visited = {title}
	final = set()
	count = 0
	pbar = tqdm(total=corpus_size, desc='1/5 wiki_cat_search')
	while queue and count<corpus_size:
		curpage = queue.pop(0)
		visited.add(curpage)
		members = wikiminer.page(curpage).categorymembers
		for member in members:
			if member not in visited:
				if "Category:" in member:
					queue.append(member)
				elif not any(exc in member for exc in exc_types) and member not in final:
					final.add(member)
					pbar.update(1)
					count+=1

	pbar.close()
	return final


def cleanChinkWikiPage(page_obj):
	page_obj['text'] = cleanText(page_obj['text'])
	filt_kps = {}
	for KP,URL in page_obj['kps'].items():
		condition_dict = {
		'found_in_text': KP in page_obj['text'],
		'valid_ngram': not (KP in ngram_set),
		'valid_charnum': len(KP)<=40,
		'valid_tokennum': len(KP.split())<=5,
		'contains_alpha': any(c.isalpha() for c in KP),
		'not_location_person': not (KP in location_set),
		'not_person': not (KP in person_set)
		}

		if all(condition_dict.keys()):
			filt_kps[KP] = URL

	page_obj['kps'] = filt_kps
	return page_obj

def wikiPageMineWorker(title):
	try:
		curpage = wikipedia.page(title, auto_suggest=False)
		soup = bs(curpage.html(), "html.parser")
		text = ' '.join([p.text for p in soup.findAll('p')])
		url = curpage.url[curpage.url.find('/wiki'):]
		kps = {}
		for kw in soup.find_all('a', href=True):
			if "/wiki/" in kw['href'] and kw.text.strip():
				kps[kw.text.strip()] = kw['href']
		return cleanChinkWikiPage({'title':title,'url':url,'text':text, 'kps':kps})
	except (ConnectionError,DisambiguationError,PageError):
		return None


wiki_pageset = wikiTitleMiner(CONFIG['wiki_category'], CONFIG['corpus_size'])
with concurrent.futures.ProcessPoolExecutor() as executor:
	corpus = list(tqdm(executor.map(wikiPageMineWorker, wiki_pageset), total=len(wiki_pageset), desc = '2/5 mine_wiki_pages'))
corpus = [ele for ele in corpus if ele]
corpus = corpus[:min(CONFIG['corpus_size'],len(corpus))]
corpus = {page['title']:dict(itertools.islice(page.items(),1,None)) for page in corpus}

###########################################################################################
###########################################################################################
#kp_pool
###########################################################################################
###########################################################################################


def textToNgramSet(text):
	ngram_set = set()
	for sent in nltk.sent_tokenize(text):
		tokens = [token for token in nltk.word_tokenize(sent) if token != ""]
		for n in range(5,-1,-1):
			ngrams = zip(*[tokens[i:] for i in range(n)])
			ngram_set.update([" ".join(ngram) for ngram in ngrams])
	return ngram_set


def minSdowDistance(orig,url_variants):
	final = []
	minSoFar = sys.maxsize
	for sublink in list(url_variants):
		temp_target = urllib.parse.unquote(sublink.replace('/wiki/',''))
		sdow_dist = sdow.sdowDistance(orig,temp_target)
		if sdow_dist and sdow_dist < minSoFar:
			minSoFar = sdow_dist
	if minSoFar == sys.maxsize:
		return -1
	return minSoFar


def kpMetricWorker(page_obj):
	global kp_pool
	final = {}
	set_text = textToNgramSet(page_obj[1]['text'])
	kps = page_obj[1]['kps']
	for kp in kp_pool.keys():
		if kp not in kps and kp in set_text:
			final[kp] = False
		elif kp in kps and kp in set_text:
			final[kp] = True
	return page_obj[0], final

def sdowDistWorker(wiki_obj):
	k = wiki_obj[0]
	v = wiki_obj[1]
	for m in v['MTCHS'].keys():
		v['MTCHS'][m] = minSdowDistance(m,v['UURLS'])
	del v['AURLS']
	return k,v




kp_pool = {}
for key in corpus:
	for k,v in corpus[key]['kps'].items():
		if k not in kp_pool:
			kp_pool[k] = {'MF':0,'HF':0,'HA':0,'HT':0,'UURLS':set(),'AURLS':[],'MTCHS':{}}
		kp_pool[k]['UURLS'].add(v)
		kp_pool[k]['AURLS'].append(v)
		kp_pool[k]['HF'] += 1
	corpus[key]['kps'] = set(corpus[key]['kps'].keys())



with concurrent.futures.ProcessPoolExecutor() as executor:
	temp_matches = list(tqdm(executor.map(kpMetricWorker, corpus.items()), total=len(corpus.items()), desc='3/5 compute_base_metrics'))
for wiki_page,add_kps in temp_matches:
	for kp,_bool in add_kps.items():
		if _bool == False:
			kp_pool[kp]['MTCHS'][wiki_page] = 0
		kp_pool[kp]['MF']+=1

del temp_matches


temp_kp_pool = {}
for k,v in kp_pool.items():
	if v['MF']>0:
		v['HT'] = round(v['HF']/v['MF'], 2) 
	if v['AURLS']:
		v['HA'] = v['AURLS'].count(max(set(v['AURLS']),key=v['AURLS'].count))/len(v['AURLS'])
	kp_pool[k] = v
	if v['MF']>=CONFIG['metric']['MF'] and v['HA']>=CONFIG['metric']['HA'] and v['HT']>=CONFIG['metric']['HA']:
		temp_kp_pool[k] = v



print(len(temp_kp_pool),'/',len(kp_pool),'remain')
kp_pool = temp_kp_pool
del temp_kp_pool

# temparr = []
# for k,v in kp_pool.items():
    # [temparr.append({'orig':k,'dest':ele}) for ele in v['MTCHS'].keys()]
# pd.DataFrame(temparr).to_csv('test_sdow_titles.csv', index=False)
# del temparr
 
if bool(CONFIG['sdow_enabled']):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		temp_kp_pool = list(tqdm(executor.map(sdowDistWorker, kp_pool.items()), total=len(kp_pool.items()), desc = '4/5 compute_sdow_metrics'))
	for obj in temp_kp_pool:
		kp_pool[obj[0]] = obj[1]
	del temp_kp_pool
else:
	print('stage 4/5 sdow_metrics skipped.')




for kw,meta in tqdm(kp_pool.items()):
	for page,dist in meta['MTCHS'].items():
		if dist>=0 and dist<CONFIG['metric']['SDOW']:
			corpus[page]['kps'].add(kw)

del kp_pool




###########################################################################################
###########################################################################################
#biogen
###########################################################################################
###########################################################################################

def biogenWorker(wiki_obj):
	wiki_obj = wiki_obj[1]
	final = []
	joinstr = '|\\|'
	kps = [kp.lower() for kp in sorted(wiki_obj['kps'], key=len, reverse=True)]

	sub_text = wiki_obj['text']
	kp_replace_patterns = sum([list(set(re.findall(re.escape(kp),sub_text,flags=re.IGNORECASE))) for kp in kps if len(kp)>=2], [])
	for pattern in kp_replace_patterns:
		sub_text = sub_text.replace(pattern,pattern.replace(' ',joinstr))

	tokens = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(sub_text)]
	for s,sent in enumerate(tokens):
		temp = []
		for t,tup in enumerate(sent):
			phrase = tup[0]
			pos = tup[1]
			phrase_split = phrase.split(joinstr)
			if len(phrase_split)>1:
				temp.append({'token':phrase_split[0], 'pos':'PH', 'bio':'B'})
				[temp.append({'token':word, 'pos':'PH', 'bio':'I'}) for word in phrase_split[1:]]
			elif phrase.lower() in kps:
				temp.append({'token':phrase, 'pos':pos, 'bio':'B'})
			else:
				temp.append({'token':phrase, 'pos':pos, 'bio':'O'})
		final.append(temp)

	return final


with concurrent.futures.ProcessPoolExecutor() as executor:
	biogen = sum(list(tqdm(executor.map(biogenWorker, corpus.items()), total=len(corpus.items()), desc='5/5 biogen')),[])



pickle.dump(biogen, open(data_loc+all_config["corpus_outfile"], "wb"))
