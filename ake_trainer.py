import json
all_config = json.load(open("config.json", "r"))
CONFIG = all_config['trainer']
data_loc = all_config['data_loc']
corpus_filename = all_config['corpus_outfile']

import pickle
import transformers as tf
import datasets as ds
import torch
import numpy as np
from tqdm import tqdm

if not torch.cuda.is_available():
	print('CUDA unavailable; training requires GPU.')
	exit()

model_checkpoint = CONFIG['model']
tokenizer = tf.AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, tf.PreTrainedTokenizerFast)

pos_ref_dict = {pos:i for i,pos in enumerate(ds.load_dataset("conll2003")["train"].features[f"pos_tags"].feature.names)}
pos_ref_dict['PH'] = len(pos_ref_dict)
bio_ref_dict = {'O':0,'B':1,'I':2}


def biogenReformat(data):
	final = {'id':[],'tokens':[],'pos_tags':[],'bio_tags':[]}
	for _id,page in tqdm(enumerate(data), desc='bio_reformat'):
		final['id'].append(_id)
		final['tokens'].append([ele['token'] for ele in page])
		final['pos_tags'].append([pos_ref_dict[ele['pos']] for ele in page])
		final['bio_tags'].append([bio_ref_dict[ele['bio']] for ele in page])
	return final


def BERTTokenize(data):
	label_all_tokens = True
	tokenized_inputs = tokenizer(data["tokens"], truncation=True, is_split_into_words=True)
	labels = []
	for i, label in enumerate(data[f"bio_tags"]):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		previous_word_idx = None
		label_ids = []
		for word_idx in word_ids:
			if word_idx is None:
				label_ids.append(-100)
			elif word_idx != previous_word_idx:
				label_ids.append(label[word_idx])
			else:
				label_ids.append(label[word_idx] if label_all_tokens else -100)
			previous_word_idx = word_idx
		labels.append(label_ids)
	tokenized_inputs["labels"] = labels
	return tokenized_inputs


corpus = pickle.load(open(data_loc+corpus_filename, "rb"))
dataset = ds.Dataset.from_dict(biogenReformat(corpus))
dataset.features['pos_tags'] = ds.Sequence(feature=ds.ClassLabel(num_classes=len(pos_ref_dict.keys()), names=list(pos_ref_dict.keys())))
dataset.features['bio_tags'] = ds.Sequence(feature=ds.ClassLabel(num_classes=len(bio_ref_dict.keys()), names=list(bio_ref_dict.keys())))
dataset = dataset.map(BERTTokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2, train_size=0.8)

label_list = dataset['train'].features[f"bio_tags"].feature.names
model = tf.AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

data_collator = tf.DataCollatorForTokenClassification(tokenizer)
metric = ds.load_metric("seqeval")



def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

print(data_loc)


args = tf.TrainingArguments(
    CONFIG['finetuned_model_name'],
    # output_dir = './model/',
    evaluation_strategy = CONFIG['train_args']['evaluation_strategy'],
    logging_strategy = CONFIG['train_args']['logging_strategy'],
    save_strategy = CONFIG['train_args']['save_strategy'],
    learning_rate=CONFIG['train_args']['learning_rate'],
    per_device_train_batch_size=CONFIG['train_args']['per_device_train_batch_size'],
    per_device_eval_batch_size=CONFIG['train_args']['per_device_eval_batch_size'],
    num_train_epochs=CONFIG['train_args']['num_train_epochs'],
    weight_decay=CONFIG['train_args']['weight_decay'],
    push_to_hub=False,
)


trainer = tf.Trainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
