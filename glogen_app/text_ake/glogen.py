import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
import re
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from nltk.tokenize.treebank import TreebankWordDetokenizer



def remove_grammar(text_to_clean):
    final = ''
    for i in text_to_clean:
        if i.isalpha() or i.isdigit() or i in ['-',' ','.']:
            final += i
    return final



def remove_bracket_content(text_to_clean):
    return re.sub("[\[\(].*?[\]\)]", "", text_to_clean)



def fix_nospace_sents(text_to_clean):
    final = text_to_clean
    for i in range(len(final)-2):
        if (final[i].isalpha() and final[i].islower()) and final[i+1] == '.' and (final[i+2].isalpha() and final[i+2].isupper()):
            final = final[:i+2] + ' ' + final[i+2:]
    return final




def print_settings():
    print("""
        1) remove_bracket_content_bool
        2) remove_grammar_bool
        3) remove_stopword_bool
        4) lemmatize_bool
        5) lowercase_bool
        """
    )




def cleanText(text_to_clean, remove_bracket_content_bool=False, remove_grammar_bool=False, remove_stopword_bool=False, lemmatize_bool=False, lowercase_bool=False):

    text_to_clean = text_to_clean.replace('\n',' ').replace('"',"'").replace('=','')
    
    if remove_bracket_content_bool == True:
        text_to_clean = remove_bracket_content(text_to_clean)
    
    if remove_grammar_bool == True:
        text_to_clean = remove_grammar(text_to_clean)
        
    clean_sent_arr = []
        
    for sent in nltk.sent_tokenize(text_to_clean):
        
        temp_sent = []
        
        if lowercase_bool == True:
            sent = sent.lower()
        
        if remove_stopword_bool == True and lemmatize_bool == True:
            for word in nltk.word_tokenize(sent):   
                if word.lower() not in stop_words:
                    temp_sent.append(lemmatizer.lemmatize(word))
                    
        if remove_stopword_bool == False and lemmatize_bool == False:
            for word in nltk.word_tokenize(sent):   
                    temp_sent.append(word)

        elif remove_stopword_bool == False and lemmatize_bool == True:
            for word in nltk.word_tokenize(sent):   
                    temp_sent.append(lemmatizer.lemmatize(word))
                    
        elif remove_stopword_bool == True and lemmatize_bool == False:
            for word in nltk.word_tokenize(sent):   
                if word.lower() not in stop_words:
                    temp_sent.append(word) 
                    
        clean_sent_arr.append(TreebankWordDetokenizer().detokenize(temp_sent))
        
    final_clean_text = ' '.join(clean_sent_arr)
    final_clean_text = re.sub(' +', ' ', final_clean_text).strip()
    final_clean_text = fix_nospace_sents(final_clean_text)

    return final_clean_text




data_loc = '/home/jackragless/projects/github/GloGen-GLOssary-GENerator/data/distilbert-base-cased-finetuned-doctrine50k/'
model_dirname = 'checkpoint-318480/'


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
 'WP': 'pronoun', 'WP$': 'pronoun', 'WRB': 'adverb', '#':'punct',
 '$':'punct', '“':'punct', '``':'punct', '(':'punct', ')':'punct',
 ',':'punct', ':':'punct',"''":'punct','.':'punct'}

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
                    pos = penn2wikt[pos_arr[0]]
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
        return {k:v for k,v in final.items() if v not in err_msg}

    return final


def glogenIterator(pdf_dict):
    final = {}
    for k,v in pdf_dict.items():
        final[k] = glogen(v)
    return final
