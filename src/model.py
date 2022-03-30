import logging
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from utils import common_args
from loadData import load_formated_dataframe, load_train_val_dataframe, load_formated_dataframe_masked_cue_bioscope, load_formated_chia_dataframe
import model_utils
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, classification_report
import torch
from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "roberta_prompt": (
        RobertaConfig,
        model_utils.RERobertaPrompt,
        RobertaTokenizer
    ),
    "bert_prompt": (
        BertConfig,
        model_utils.REBertPrompt,
        BertTokenizer
    )
}

DATA_CLASS = {
    'i2b2': '/i2b2/test.csv',
    'bioscope': '/bioscope/clinical-bioc.csv',
    'negex': '/negex/negex.csv',
    'i2b2-2012': '/i2b2_2012/test.csv',
    'chia-balanced': '/chia/chia_without_scope_balanced.csv',
    'mimic-iii': '/mimic-clinical-assertion/mimic_iii_assertions.csv', 
    'mimic-discharge': '/mimic-clinical-assertion/discharge_summaries_assertions.csv',
    'mimic-nursing': '/mimic-clinical-assertion/nursing_assertions.csv',
    'mimic-physician': '/mimic-clinical-assertion/physician_assertions.csv',
    'mimic-radiology': '/mimic-clinical-assertion/radiology_assertions.csv'
}

DATA_DIR = '/home/sw37643/Negation_Detection/Data'
MODEL_DIR = '/home/sw37643/Negation_Detection/Prompt_baseline/models'
#LABEL_OF_INTEREST = ['P', 'N', 'U', 'H', 'C', 'O']
LABEL_OF_INTEREST = ['P', 'N', 'U']

EPOCHS = 20
LR = 1e-6 #5e-7

def plot_multiclass_confusion_matrix(pred, true):
    
    print(f'precision: {precision_score(true, pred, average="micro", zero_division=0): .3f}, \
        recall: {recall_score(true, pred, average="micro", zero_division=0): .3f}, \
        f1: {f1_score(true, pred, average="micro"): .3f}, \
        accuracy: {accuracy_score(true, pred): .3f}')

    print('------------------------------------------')
    # 'P': 0, 'N': 1, 'U': 2, 'H': 3, 'C': 4, 'O': 5
    #print(multilabel_confusion_matrix(true, pred, labels=[0, 1, 2, 3, 4, 5]))
    print(multilabel_confusion_matrix(true, pred, labels=[0, 1, 2]))
    print(classification_report(true, pred, digits=4))

    #return multilabel_confusion_matrix(true, pred, labels=[0, 1, 2, 3, 4, 5]), classification_report(true, pred, digits=4)
    return multilabel_confusion_matrix(true, pred, labels=[0, 1, 2]), classification_report(true, pred, digits=4)

def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, criterion, epoch, num_epochs):
    eval_output_dir = args.eval_output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    count = 0
    epoch_loss = 0.0
    
    pred_list, true_list = [], []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            if args.use_prompt:
                input_ids, attention_mask, labels, mask_pos = batch[0], batch[1], batch[2], batch[3]
                inputs = {'input_ids': input_ids,
                         'attention_mask': attention_mask,
                         'labels': labels}
            
                logits, loss = model(**inputs)

                #from_answer_to_index, from_index_to_answer, ground_truth_dict = model_utils.from_answer_to_dicts(['P', 'N', 'U', 'H', 'C', 'O'], tokenizer)
                from_answer_to_index, from_index_to_answer, ground_truth_dict = model_utils.from_answer_to_dicts(['P', 'N', 'U'], tokenizer)

                assert all([val in list(from_index_to_answer.keys()) for val in torch.gather(labels, 1, mask_pos.unsqueeze(-1)).view(-1)])

                mask_pos_formatted = mask_pos.unsqueeze(-1).repeat(1, tokenizer.vocab_size).unsqueeze(1) 
                logits = torch.gather(logits, 1, mask_pos_formatted) 
                logits = logits[:, :, list(from_index_to_answer.keys())].squeeze(1)  
                truth = torch.tensor([ground_truth_dict[val] for val in torch.gather(labels, 1, mask_pos.unsqueeze(-1)).view(-1).tolist()])
            
            else:
                input_ids, attention_mask, truth = batch[0], batch[1], batch[2]
                inputs = {'input_ids': input_ids,
                        'attention_mask': attention_mask}

                logits = model(**inputs)
                loss = criterion(logits, truth)

            pred_softmax = torch.nn.Softmax(1)(logits).cpu().detach().numpy()
            pred = torch.from_numpy(np.argmax(pred_softmax, axis=1))
            
            for v in pred.cpu():
                pred_list.append(v)
            for v in truth.cpu():
                true_list.append(v)
                
            count += int(torch.sum(pred == truth.cpu()))

            epoch_loss += loss.item() * len(truth)
    
    epoch_loss = epoch_loss / len(eval_dataset)
    epoch_acc = count / len(eval_dataset)

    print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f}'.format(
          epoch + 1, num_epochs, "eval", epoch_loss, epoch_acc))

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        
    confusion_matrix, report = plot_multiclass_confusion_matrix(pred_list, true_list)
    
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write('Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f}'.format(
                     epoch + 1, num_epochs, "eval", epoch_loss, epoch_acc))
        writer.write('\n')
        writer.write('----------- | {:^5} | Confusion Matrix: {} | Report: {} '.format(
                     "eval", confusion_matrix, report))
        writer.write('\n')

    return epoch_acc

def train(args, train_dataset, eval_dataset, external_eval_datasets, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, criterion):
    eval_output_dir = args.eval_output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        num_train_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        num_train_steps = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_steps
    )

    model.zero_grad()
    seed_everything(args)

    es = model_utils.EarlyStopping(patience=args.patience)

    torch.cuda.empty_cache()
    for epoch in tqdm(range(0, int(args.num_train_epochs))):
        count = 0
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            optimizer.zero_grad()

            if args.use_prompt:
                input_ids, attention_mask, labels, mask_pos = batch[0], batch[1], batch[2], batch[3]
                inputs = {'input_ids': input_ids,
                         'attention_mask': attention_mask,
                         'labels': labels}
            
                logits, loss = model(**inputs)

                #from_answer_to_index, from_index_to_answer, ground_truth_dict = model_utils.from_answer_to_dicts(['P', 'N', 'U', 'H', 'C', 'O'], tokenizer)
                from_answer_to_index, from_index_to_answer, ground_truth_dict = model_utils.from_answer_to_dicts(['P', 'N', 'U'], tokenizer)

                # assert we retrieve logits at <mask> position
                assert all([val in list(from_index_to_answer.keys()) for val in torch.gather(labels, 1, mask_pos.unsqueeze(-1)).view(-1)])
                
                # (batch_size, 1, vocab_size)
                mask_pos_formatted = mask_pos.unsqueeze(-1).repeat(1, tokenizer.vocab_size).unsqueeze(1) 
                logits = torch.gather(logits, 1, mask_pos_formatted) 
                logits = logits[:, :, list(from_index_to_answer.keys())].squeeze(1)  
                truth = torch.tensor([ground_truth_dict[val] for val in torch.gather(labels, 1, mask_pos.unsqueeze(-1)).view(-1).tolist()])
            
            else:
                input_ids, attention_mask, truth = batch[0], batch[1], batch[2]
                inputs = {'input_ids': input_ids,
                        'attention_mask': attention_mask}

                logits = model(**inputs)
                loss = criterion(logits, truth)

            pred_softmax = torch.nn.Softmax(-1)(logits).cpu().detach().numpy()
            pred = torch.from_numpy(np.argmax(pred_softmax, axis=1))
            count += int(torch.sum(pred == truth.cpu()))
            epoch_loss += loss.item() * len(truth)

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        epoch_loss = epoch_loss / len(train_dataset)
        epoch_acc = count / len(train_dataset)

        print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f}'.format(
            epoch + 1, args.num_train_epochs, "train", epoch_loss, epoch_acc))

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            writer.write('Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f}'.format(
                        epoch + 1, args.num_train_epochs, "train", epoch_loss, epoch_acc))
            writer.write('\n')
            
        print('----Evaluate on I2B2----')
        eval_epoch_acc = evaluate(args, eval_dataset, model, tokenizer, criterion,
                                  epoch=epoch, num_epochs=int(args.num_train_epochs))
        
        if len(external_eval_datasets) == 0:
            print('skip evaluating other datasets.')
        elif len(external_eval_datasets) != 1:
            print('----External Evaluate on I2B2 2012----')
            external_eval_epoch_acc = evaluate(args, external_eval_datasets[0], model, tokenizer, criterion,
                                      epoch=epoch, num_epochs=int(args.num_train_epochs))
            
            print('----External Evaluate on MIMIC-III ----')
            external_eval_epoch_acc = evaluate(args, external_eval_datasets[1], model, tokenizer, criterion,
                                      epoch=epoch, num_epochs=int(args.num_train_epochs))

            print('----External Evaluate on BioScope----')
            external_eval_epoch_acc = evaluate(args, external_eval_datasets[2], model, tokenizer, criterion,
                                      epoch=epoch, num_epochs=int(args.num_train_epochs))
            
            print('----External Evaluate on NegEx----')
            external_eval_epoch_acc = evaluate(args, external_eval_datasets[3], model, tokenizer, criterion,
                                      epoch=epoch, num_epochs=int(args.num_train_epochs))
            
            print('----External Evaluate on Chia-balanced ----')
            external_eval_epoch_acc = evaluate_error_analysis(args, external_eval_datasets[4], model, tokenizer, criterion,
                                      epoch=epoch, num_epochs=int(args.num_train_epochs))
            
        
        es(eval_epoch_acc, model, tokenizer, save_to_path=args.params_output_dir)
        
        #torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'loss': epoch_loss,
        #    }, f'{MODEL_DIR}/Prompt_bert_checkpoint_epoch{epoch}_testAccuracy_{es.show_best_acc():.3f}.pt')

        if es.early_stop:
            print(f"Best Accuracy: {es.show_best_acc():.4f}")
            break

        if epoch == args.num_train_epochs - 1:
            print(f"Best Accuray: {es.show_best_acc():.4f}")

    with open(output_eval_file, "a") as writer:
        writer.write(f"Best Accuracy: {es.show_best_acc():.4f}, Model: {args.model_type}")
        writer.write('\n\n')

        
def load_evaluate_data(name):
    
    external_test = pd.read_csv(DATA_DIR + DATA_CLASS[name])
    
    if name=='i2b2-2012':
        external_test.dropna(subset = ["label"], inplace=True)
        external_test = external_test[external_test['label'].isin(LABEL_OF_INTEREST)]
        
    if name=='bioscope':
        # remove None concepts
        external_test.dropna(subset = ["concept"], inplace=True)
        
    if name=='negex':
        external_test = external_test[external_test['concept_start']!=-1]
    
    external_test_df = load_formated_dataframe(external_test)
    external_val_features = model_utils.convert_examples_to_features(external_test_df, tokenizer, args.max_input_length, args.use_prompt)
    external_eval_dataset = model_utils.convert_features_to_dataset(external_val_features)
    
    return external_eval_dataset

def load_chia_evaluation(name):
    external_test = pd.read_csv(DATA_DIR + DATA_CLASS[name])

    if 'chia' in name:
        external_test = external_test.replace({np.nan: None})

    external_test_df = load_formated_chia_dataframe(external_test)
    external_val_features = model_utils.convert_chia_examples_to_features(external_test_df, tokenizer, args.max_input_length, args.use_prompt)
    external_eval_dataset = model_utils.convert_features_to_dataset(external_val_features)
    
    return external_eval_dataset

if __name__ == "__main__":

    parser = common_args()
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu_device)
    args.device = device
    seed_everything(args)

    if (os.path.exists(args.params_output_dir)
        and os.listdir(args.params_output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overwrite.".format(
                args.params_output_dir
            )
        )

    if not os.path.exists(args.params_output_dir):
        os.makedirs(args.params_output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.params_output_dir, 'model.log')
    )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    criterion = model_utils.loss_fn

    if args.input_dir is not None:
        print(f'loading model from {args.input_dir}')
        config = config_class.from_pretrained(args.input_dir)
        tokenizer = tokenizer_class.from_pretrained(args.input_dir)
        model = model_class(config, args.input_dir)
        assert tokenizer.special_tokens_map['additional_special_tokens'] == 2
    else:
        print('loading pre-trained model')
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        model = model_class(config=config, parameters=args.model_name_or_path)
        
        special_tokens_dict = {'additional_special_tokens': ['<E>','</E>']}
        
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 2

        model.resize_token_embeddings(len(tokenizer))
        model.embeddings.word_embeddings.padding_idx = 1

        # Temporarily disabling gradient calculation. (Initializing weights is not an operation that needs to be accounted for in backpropagation)
        with torch.no_grad():
            model.embeddings.word_embeddings.weight[-3:, :] = torch.nn.init.xavier_uniform_(torch.zeros([3, model.config.hidden_size]))

    model.to(args.device)
    
    # load i2b2 training data
    beth_df = pd.read_csv(DATA_DIR + '/i2b2/beth.csv')
    partners_df = pd.read_csv(DATA_DIR + '/i2b2/partners.csv')
    df_test = pd.read_csv(DATA_DIR + '/i2b2/test.csv')
    
    df_train = pd.concat([beth_df, partners_df], ignore_index=True)
    df_train = df_train[df_train['label'].isin(LABEL_OF_INTEREST)]
    df_test = df_test[df_test['label'].isin(LABEL_OF_INTEREST)]
    
    train_df = load_formated_dataframe(df_train)
    test_df = load_formated_dataframe(df_test)
    
    train_features= model_utils.convert_examples_to_features(train_df, tokenizer, args.max_input_length, args.use_prompt)
    val_features= model_utils.convert_examples_to_features(test_df, tokenizer, args.max_input_length, args.use_prompt)
    
    eval_dataset= model_utils.convert_features_to_dataset(val_features)
    
    ##
    external_eval_dataset_0 = load_evaluate_data('i2b2-2012')
    external_eval_dataset_1 = load_evaluate_data('mimic-iii')
    external_eval_dataset_2 = load_evaluate_data('bioscope')
    external_eval_dataset_3 = load_evaluate_data('negex')
    external_eval_dataset_4 = load_chia_evaluation('chia-balanced')

    external_eval_datasets = [external_eval_dataset_0, external_eval_dataset_1, external_eval_dataset_2, external_eval_dataset_3, external_eval_dataset_4]
    
    if args.do_eval:
        print("Start Evaluation.")
        evaluate(args, external_eval_dataset, model, tokenizer,
                 criterion, epoch=0, num_epochs=0)
        logger.info("Evaluation parameters %s", args)

    if args.do_train:
        logger.info("Training parameters %s", args)
        train_dataset= model_utils.convert_features_to_dataset(train_features)
        # 6 class
        #train(args, train_dataset, eval_dataset, [], model, tokenizer, criterion)

        # 3 class
        train(args, train_dataset, eval_dataset, external_eval_datasets, model, tokenizer, criterion)
        
        
