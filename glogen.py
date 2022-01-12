import json
CONFIG = json.load(open("config.json", "r"))
data_loc = CONFIG['data_loc']
model_dirname = CONFIG['glogen']['model_dirname']
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
if not os.path.isdir(data_loc+model_dirname):
	print('Model directory does not exist!')
	exit()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', str(device).upper())


from tqdm import tqdm
import transformers
import nltk
from scipy.special import softmax
import numpy as np
from text_cleaner import cleanText
import wikipedia 
from wiktionaryparser import WiktionaryParser
parser = WiktionaryParser()
parser.set_default_language('english')
from sentence_transformers import SentenceTransformer, util
semsim_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from Levenshtein import ratio as lev
from bs4 import GuessedAtParserWarning
import warnings
warnings.filterwarnings('ignore', category=GuessedAtParserWarning)


err_msg = ['POS NOT FOUND.', 'KP NOT FOUND.']

penn2wikt = {'CC': 'conjunction', 'CD': 'numeral', 'DT': 'determiner',
'EX': 'predicative', 'FW': 'noun', 'IN': 'preposition', 'JJ': 'adjective', 
'JJR': 'adjective', 'JJS': 'adjective', 'LS': 'symbol', 'MD': 'letter', 
'NN': 'noun', 'NNS': 'noun', 'NNP': 'noun', 'NNPS': 'noun',
 'PDT': 'predicative', 'POS': 'pronoun', 'PRP': 'pronoun', 
 'PRP$': 'pronoun', 'RB': 'adverb', 'RBR': 'adverb', 'RBS': 'adverb', 
 'RP': 'particle', 'SYM': 'symbol', 'TO': 'preposition', 'UH': 'interjection', 
 'VB': 'verb', 'VBD': 'verb', 'VBG': 'participle', 'VBN': 'participle', 
 'VBP': 'verb', 'VBZ': 'verb', 'WDT': 'determiner', 
 'WP': 'pronoun', 'WP$': 'pronoun', 'WRB': 'adverb'}

tokenizer = transformers.AutoTokenizer.from_pretrained(data_loc+model_dirname)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
ake_model = transformers.AutoModelForTokenClassification.from_pretrained(data_loc+model_dirname)



def attentionMatrixToSoftmax(input_ids, probs):
	final = []
	id_to_label = {0:'O',1:'B',2:'I'}
	for i,word in enumerate(probs[0][0]):
		final.append((tokenizer.decode(input_ids[0][i]),{id_to_label[i]:prob for i,prob in enumerate(softmax([prob.item() for prob in word]))}))
	return final


def BERTPredict(text):
	#predictions
	global ake_model
	tok_sent = tokenizer(text, return_tensors="pt", truncation=True)
	attmat = ake_model(tok_sent['input_ids'],attention_mask=tok_sent['attention_mask'])
	softmax = attentionMatrixToSoftmax(tok_sent['input_ids'], attmat)
	
	#reconnect hash-split tokens
	final = []
	recontok = ''
	metrics = {'B':[],'I':[],'O':[]}
	for i in range(len(softmax)):
		if softmax[i][0] in ['[CLS]','[SEP]']:
			continue
		recontok += softmax[i][0]
		[metrics[k].append(v) for k,v in softmax[i][1].items()]
		if not softmax[i+1][0].startswith('##'):
			recontok = recontok.replace('##','')
			metrics = {k:np.average(np.array(v)) for k,v in metrics.items()}
			final.append([recontok,metrics])
			recontok = ''
			metrics = {'B':[],'I':[],'O':[]}
	
	return final

def KPParse(bert_preds, lev_ratio):
	tempKP = ''
	avgcon = []
	final = []
	seenSoFar = set()
	pos_arr = []
	pos_tags = nltk.pos_tag([ele[0] for ele in bert_preds])
	for i,token_obj in enumerate(bert_preds):
		if max(token_obj[1], key=token_obj[1].get) != 'O':
			tempKP += token_obj[0] + ' '
			avgcon.append(token_obj[1][max(token_obj[1], key=token_obj[1].get)])
			pos_arr.append(pos_tags[i][1])
		else:
			if tempKP:
				if len(pos_arr)==1:
					if pos in penn2wikt.keys():
						pos = penn2wikt[pos_arr[0]]
					else:
						pos = 'other'
				elif 'VB' in pos_arr:
					pos = 'verb'
				else:
					pos = 'noun'
				if not seenSoFar or max([lev(tempKP.strip().lower(),ele) for ele in list(seenSoFar)])<lev_ratio:
					final.append([tempKP.strip(), pos, np.average(np.array(avgcon))])
					seenSoFar.add(tempKP.strip().lower())
			tempKP = ''
			avgcon = []
			pos_arr = []
			
	final.sort(key=lambda x:x[2], reverse=True)
	return {ele[0]:ele[1] for ele in final}
		
	
def topKPs(text, lev_ratio=0.85):
	final = set()
	combined = []
	for sent in tqdm(nltk.sent_tokenize(text), desc='AKE'):
		combined.extend(BERTPredict(sent))
	return KPParse(combined, lev_ratio)


def wiktDictSimplify(wikt_dict):
	final = {}
	for ety in wikt_dict:
		for text in ety['definitions']:
			pos = text['partOfSpeech'].lower().replace('proper ','')
			if pos not in final.keys():
				final[pos] = []
			for _def in text['text'][1:]:
				if _def and _def[0]=='(':
					final[pos].append({'cat':_def[1:_def.find(')')].strip().lower(),'def':_def[_def.find(')')+1:].strip()})
				else:
					final[pos].append({'cat':None,'def':_def.strip()})
	return final


def wiktPredictBERT(def_candidates, source_text):

	global semsim_model
	
	clean_sents = []
	orig_sents = []
	for obj in def_candidates:
		orig_sents.append(obj)
		if obj['cat']:
			clean_sents.append(cleanText(obj['cat']+' '+obj['def'], True, True, True, True, True))
		else:
			clean_sents.append(cleanText(obj['def'], True, True, True, True, True))
	
	embeddings1 = semsim_model.encode(clean_sents, convert_to_tensor=True)
	embeddings2 = semsim_model.encode([source_text], convert_to_tensor=True)
	cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
	scores_arr = [ele.item() for ele in list(cosine_scores)]
	
	return orig_sents[scores_arr.index(max(scores_arr))]['def']


def wikiDefgen(KP, err_msg='Wikipedia page not found!'):
	try:
		wiki_def = wikipedia.summary(KP,auto_suggest=False)
		return wiki_def.split('.')[0]+'.'
	except:
		pass
	return err_msg	


def definitionCandidateGenerator(KP, pos):
	wikt_query = parser.fetch(KP)

	if not wikt_query or not wikt_query[0]['definitions']:
		if any([grammar in KP for grammar in [' - '," ' "]]):
			return definitionCandidateGenerator(KP.replace(' - ','-').replace(" ' ","'"), pos)
		else:
			return KP, [{'cat':None, 'def':wikiDefgen(KP,err_msg[1])}]

	wikt_query = wiktDictSimplify(wikt_query)
	if pos in wikt_query.keys():
		return KP, wikt_query[pos]
	else:
		return KP, sum(wikt_query.values(),[])



def glogen(source_text, see_missing=False):
	requery_arr = ['Initialism of', 'plural of', 'genitive singular of', 'Short for']
	pattern = '|'.join(requery_arr)

	KPs = topKPs(cleanText(source_text))

	final = {}
	with ProcessPoolExecutor() as executor:
		candidates = list(tqdm(executor.map(definitionCandidateGenerator,[ele[0] for ele in KPs.items()],[ele[1] for ele in KPs.items()]), total=len(KPs), desc='defgen'))

	for cnd in candidates:
		final[cnd[0]] = wiktPredictBERT(cnd[1], source_text)
		if cnd:
			final[cnd[0]] = wiktPredictBERT(cnd[1], source_text)
			if any([final[cnd[0]].startswith(substr) for substr in requery_arr]):
				temp = re.sub(pattern, '', final[cnd[0]]).strip().replace('.','')
				final[cnd[0]] += '; ' + wiktPredictBERT(definitionCandidateGenerator(temp,'')[1], source_text)

	if not see_missing:
		return {k:final[k].replace('\n','') for k in sorted(final.keys()) if final[k] not in err_msg}

	return final