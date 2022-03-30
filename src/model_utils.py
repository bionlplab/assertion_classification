import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import os
from utils import common_args
from loadData import load_formated_dataframe, load_train_val_dataframe
import random 
from transformers import (
    RobertaModel, 
    RobertaConfig, 
    RobertaForMaskedLM,
    #RobertaPreTrainedModel,
    RobertaTokenizer, 
    BertModel, 
    BertPreTrainedModel,
    BertForMaskedLM,
    BertConfig, 
    BertTokenizer, 
    PreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings


class InputFeatures(object):
    def __init__(self, text, input_ids, attention_mask, ent1, label, eid, **kwargs):
        self.text = text
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.concept = ent1
        self.label = label
        self.eid = eid

        if 'mask_pos' in kwargs:
            self.mask_pos = kwargs['mask_pos']


def from_answer_to_dicts(answers: List, tokenizer) -> Tuple[Dict, Dict, Dict]:
    from_answer_to_index = {}

    for answer in answers:
        from_answer_to_index[answer] = tokenizer.encode(answer, add_special_tokens=False)[0]
    
    from_index_to_answer = {from_answer_to_index[word]: word for word in from_answer_to_index}
    ground_truth_dict = {idx:list(from_index_to_answer.keys()).index(idx) for idx in from_index_to_answer}

    return from_answer_to_index, from_index_to_answer, ground_truth_dict


def convert_examples_to_features(df, tokenizer, max_input_length, use_prompt: bool, **kwargs):

    '''
    Convert examples to features 
    '''

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    msk_id = tokenizer.mask_token_id

    if use_prompt:
        #label_map = {"P": "P", "N": "N", "U": "U", "H": "H", "C": "C", "O": "O"}
        label_map = {"P": "P", "N": "N", "U": "U"}
        
        features = []
        for _, row in df.iterrows():

            text = str(row.text).replace(f"{row.concept}", f"<E>{row.concept}</E>")
            
            #prompt = f"<E>{row.concept}</E> is present, absent, uncertain, hypothetical, conditional or N/A? <mask>."
            prompt = f"<E>{str(row.concept)}</E> is [MASK]."
            #prompt = f"[MASK]: "
                        
            text_token_ids = tokenizer.encode(text, add_special_tokens=False)
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            avail_len = max_input_length - 3 - len(prompt_token_ids)
            input_ids = [cls_id] + text_token_ids[-avail_len:] + [sep_id] + prompt_token_ids + [sep_id]
            
            pad_len = max_input_length - len(input_ids)
            if pad_len > 0:
                input_ids += [pad_id] * pad_len
            
            mask_pos = input_ids.index(msk_id)
            
            #target = tokenizer.encode(random.sample(label_map[row.label], 1)[0], add_special_tokens=False)[0]
            target = tokenizer.encode(label_map[row.label], add_special_tokens=False)[0]
            
            label = torch.where(torch.tensor(input_ids) != msk_id, -100, target).tolist()

            features.append(InputFeatures(text=text,
                                        input_ids=input_ids,
                                        attention_mask=torch.where(torch.tensor(input_ids) != pad_id, torch.tensor(1), torch.tensor(0)).tolist(),
                                        ent1=row.concept,
                                        label=label,
                                        mask_pos=mask_pos,
                                        eid=row.id))
            
    assert len(features) == len(df)
    return features

def convert_chia_examples_to_features(df, tokenizer, max_input_length, use_prompt: bool, **kwargs):

    '''
    Convert examples to features 
    '''

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    msk_id = tokenizer.mask_token_id

    if use_prompt:
        #label_map = {"P": "P", "N": "N", "U": "U", "H": "H", "C": "C", "O": "O"}
        label_map = {"P": "P", "N": "N", "U": "U"}

        features = []
        for _, row in df.iterrows():
            
            # for chia
            text = str(row.text).replace(f"{row.text[row.concept_start: row.concept_end]}", f"<E>{row.text[row.concept_start: row.concept_end]}</E>")
            
            #prompt = f"<E>{row.concept}</E> is present, absent, uncertain, hypothetical, conditional or N/A? <mask>."
            prompt = f"<E>{row.text[row.concept_start: row.concept_end]}</E> is [MASK]."
            #prompt = f"[MASK]: "
                        
            text_token_ids = tokenizer.encode(text, add_special_tokens=False)
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            avail_len = max_input_length - 3 - len(prompt_token_ids)
            input_ids = [cls_id] + text_token_ids[-avail_len:] + [sep_id] + prompt_token_ids + [sep_id]
            
            pad_len = max_input_length - len(input_ids)
            if pad_len > 0:
                input_ids += [pad_id] * pad_len
            
            mask_pos = input_ids.index(msk_id)
            
            target = tokenizer.encode(random.sample(label_map[row.label], 1)[0], add_special_tokens=False)[0]
            #target = tokenizer.encode(label_map[row.label], add_special_tokens=False)[0]

            label = torch.where(torch.tensor(input_ids) != msk_id, -100, target).tolist()

            features.append(InputFeatures(text=text,
                                        input_ids=input_ids,
                                        attention_mask=torch.where(torch.tensor(input_ids) != pad_id, torch.tensor(1), torch.tensor(0)).tolist(),
                                        ent1=row.concept,
                                        label=label,
                                        mask_pos=mask_pos,
                                        eid=row.id))
            
    assert len(features) == len(df)
    return features


def convert_features_to_dataset(features):
    '''
    df = load_formated_dataframe([path])
    features = convert_examples_to_features(df , tokenizer, input_max_length, use_prompt)
    '''
    input_ids = torch.tensor([f.input_ids for f in features])
    attention_mask = torch.tensor([f.attention_mask for f in features])
    label = torch.tensor([f.label for f in features])

    # Check if InputFeatures contains mask_pos (for MLM)
    if hasattr(features[0], 'mask_pos'):
        mask_pos = torch.tensor([f.mask_pos for f in features])
        dataset = TensorDataset(input_ids, attention_mask, label, mask_pos)
    else:
        dataset = TensorDataset(input_ids, attention_mask, label)
    
    return dataset


class RERoberta(RobertaModel):
    def __init__(self, config, parameters, n_class=2):
        super().__init__(config)
        
        self.model = RobertaModel(config).from_pretrained(parameters)
        self.embeddings = self.model.embeddings
        self.logits = nn.Linear(self.model.config.hidden_size, n_class)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        
    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids, attention_mask)
        cls_hs = output.pooler_output
        logit = self.logits(cls_hs)  
        
        return logit


class RERobertaPrompt(RobertaModel):
    def __init__(self, config, parameters):
        super().__init__(config)

        self.model = RobertaForMaskedLM(config).from_pretrained(parameters)
        self.embeddings = self.model.roberta.embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.roberta.set_input_embeddings(value)

    def forward(self, **inputs):

        outputs = self.model(**inputs)
        logit = outputs.logits
        loss = outputs.loss
        
        return logit, loss

class REBertPrompt(BertModel):
    def __init__(self, config, parameters, n_class=3):
        super().__init__(config)
        
        self.model = BertForMaskedLM(config).from_pretrained(parameters)
        self.embeddings = self.model.bert.embeddings
        #self.logits = nn.Linear(self.model.config.hidden_size, n_class)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        
    def forward(self, **inputs):

        output = self.model(**inputs)
        logit = output.logits
        loss = output.loss
        
        return logit, loss


class REBert(BertModel):
    def __init__(self, config, parameters, n_class=3):
        super().__init__(config)
        
        self.model = BertModel(config).from_pretrained(parameters)
        self.embeddings = self.model.embeddings
        self.logits = nn.Linear(self.model.config.hidden_size, n_class)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        
    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids, attention_mask)
        cls_hs = output.pooler_output
        logit = self.logits(cls_hs)  
        
        return logit


def loss_fn(logit, label):
    # i2b2 2010 - 6 class
    normedWeights = [0.254938, 0.73861738, 3.81499461, 3.08595113, 16.14840183, 13.24531835]
    
    # i2b2 2010 - 3 class
    #normedWeights = [0.47066032, 1.36361738, 7.04314995]
    
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=normedWeights)
    #ce_loss = nn.CrossEntropyLoss(reduction='mean')
    return ce_loss(logit, label)


class EarlyStopping:
    """Early stops the training if validation loss doesn't decrease after a given patience."""
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, acc, model: PreTrainedModel, tokenizer, save_to_path):

        score = acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model, tokenizer, save_to_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc, model, tokenizer, save_to_path)
            self.counter = 0

    def save_checkpoint(self, acc, model: PreTrainedModel, tokenizer, save_to_path):
        '''Saves model when validation score improve.'''
        model.save_pretrained(save_to_path)
        torch.save(model.state_dict(), os.path.join(save_to_path, "training_args.bin"))
        tokenizer.save_pretrained(save_to_path)
        self.val_loss_min = acc
        
    def show_best_acc(self):
        return self.best_score


if __name__ == '__main__':
    parser= common_args()
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    df = load_formated_dataframe([args.BioInfer_data])

    features = convert_examples_to_features(df , tokenizer, 100)
    dataset = convert_features_to_dataset(features)

    
    