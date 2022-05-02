
from transformers import AutoTokenizer, AutoModelWithLMHead
import os
import pandas as pd
import transformers
from transformers import RobertaModel
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel
import torch
from transformers import get_scheduler
from transformers import AdamW

def candid_fine_tuning_candid(dir_base= "Z:/"):



    #tokenizer = AutoTokenizer.from_pretrained('/Users/zmh001/Documents/language_models/bert/')
    #bert = AutoModelWithLMHead.from_pretrained('/Users/zmh001/Documents/language_models/bert/')

    #tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")
    #bert = RobertaModel.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")

    language_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')

    #tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/bio_clinical_bert/", truncation=True)
    #tokenizer = AutoTokenizer.from_pretrained("Z:/Zach_Analysis/roberta/", truncation=True)
    #bert = AutoModelWithLMHead.from_pretrained("/Users/zmh001/Documents/language_models/bio_clinical_bert/")
    #bert = AutoModelWithLMHead.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")
    #bert = AutoModelWithLMHead.from_pretrained("Z:/Zach_Analysis/roberta/")

    tokenizer = AutoTokenizer.from_pretrained(language_path, truncation=True)
    bert = AutoModelWithLMHead.from_pretrained(language_path)



    reports_file = 'Pneumothorax_reports.csv'
    report_direct = os.path.join(dir_base, 'public_datasets/candid_ptx')

    model_direct = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/bert_mlm/')


    # first, get the data into correct format -- text blocks.
    text_file = reports_file.replace('.csv', '.txt')

    # make file if it doesn't exist
    if not os.path.exists(os.path.join(report_direct, text_file)):
        df_report = pd.read_csv(os.path.join(report_direct, reports_file))
        with open(os.path.join(report_direct, text_file), 'w') as w:
            for i, row in df_report.iterrows():
                entry = str(row["Report"]).replace('\n', ' ')
                w.write(entry + '\n')



    #get vocab needed to add
    #report_direct = 'Z:/Lymphoma_UW_Retrospective/Reports/'

    vocab_file = ''
    #if we want to expand vocab file
    save_name_extension = ''
    """
    if os.path.exists(os.path.join(report_direct, vocab_file)) and not vocab_file == '' :
        vocab = pd.read_csv(os.path.join(report_direct, vocab_file))
        vocab_list = vocab["Vocab"].to_list()

        print(f"Added vocab length: {str(len(vocab_list))}")
        print(f"Original tokenizer length: {str(len(tokenizer))}")

        #add vocab
        tokenizer.add_tokens(vocab_list)

        print(f"New tokenizer length: {str(len(tokenizer))}")

        #expand model
        bert.resize_token_embeddings(len(tokenizer))
        save_name_extension = '_new_vocab'
    """

    dataset = transformers.LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(report_direct, text_file),
        block_size=16
    )

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )



    training_args = transformers.TrainingArguments(
        #output_dir='/Users/zmh001/Documents/language_models/trained_models/bert_pretrained_v3/',
        output_dir=model_direct,
        overwrite_output_dir=True,
        seed = 117,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=1e-6,
        warmup_steps=10,
        save_steps=10_000,
        save_total_limit=3,
    )

    trainer = transformers.Trainer(
        model=bert,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(model_direct)




