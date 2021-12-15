# ai_preprocess_utils module centralises all text cleaning processes

import nltk
nltk.download('punkt')
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
