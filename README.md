# GloGen: _automatic domain-specific GLOssary Generator_

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

## How Does It Work?  

GloGen consists of two primary components:  
1) **Automatic keyphrase extractor (AKE)** ---> determines what to define from text.  
2) **Definition Generator (DEFGEN)** ---> sources definitions from Wiktionary.   


##### AKE Module:  
`ake_corpus.py` mines a corpus of Wikipedia pages from specific category (eg. "Category:Machine learning"): https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories  

Wikilinks within each page are used as proxy for keyphrases.  
Excerpt from BERT Wiki:  
**Bidirectional Encoder Representations from Transformers (BERT)** is a **transformer**-based 
**machine learning** technique for **natural language processing (NLP)** pre-training developed by **Google**.  

`ake_trainer.py` uses this training coprus to finetune a BERT token classification model (Huggingface Transformers).  

##### DEFGEN:
`glogen.py` requests definition variants of a given keyphrase from Wiktionary.  
It then narrows down candidate definitions using part-of-speech, compares each to the source text using semantic similarity, and predicts best definition.  
When all predictions are finished, glogen.py returns a glossary dictionary. 



##### Configuration:
Modify `config.json` to adjust variables for `ake_corpus.py` | `ake_trainer.py` | `glogen.py`.


## Check Out These Repos!
https://github.com/Suyash458/WiktionaryParser  
https://github.com/jwngr/sdow  
https://github.com/martin-majlis/Wikipedia-API  
https://github.com/huggingface/transformers  


## License:
Apache 2.0
