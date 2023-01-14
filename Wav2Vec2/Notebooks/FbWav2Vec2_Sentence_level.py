import GPUtil
import torch
import os
import paramiko, re
import sys, os, stat
import torchaudio
from paramiko import AutoAddPolicy
import pandas as pd
from datasets import Dataset, load_metric
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import torchaudio
import librosa
import numpy as np
import mlflow
import mlflow.pytorch
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import glob 
from ReleaseCache import GPUCache
import gc
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

parameters= json.load (open('/home/elias/code/model/properties.json', 'r'))
    
batch_size= parameters['training_batch_size']


#-----------------Function for loading fata from server---------------------------------------#

def start_train(list_of_wavs, last_index):
      
    list_txts = []
    
    for path in list_of_wavs:
        id = path.split('/')[-1].split('.')[0]
        folder = path.split('/')[5]
        text_path = 'PUT THE TEXT PATH'
        text = ""
        with open(text_path, "r") as f:
            text = f.readline()
        list_txts.append(text)
        # print(text_path)

    #-----------------------Use pandas to load the data as a dataframe-----------# 

    df = pd.DataFrame(list(zip(list_of_wavs, list_txts)),
                columns = ['audio_path', 'label'])

    copy_df= df.copy()
    
    


    # --------------------Train, test and split-------------------------------#

    train_len=int(batch_size*0.8)
    
    
    train_data = copy_df[:train_len]
    dev_data  = copy_df[train_len:batch_size]
    
    #test_Data = copy_df[80:]

    # print(dev_data)
    train_data = Dataset.from_pandas(train_data)
    dev_data = Dataset.from_pandas(dev_data)

    
    def extract_all_chars(batch):
        
        all_text = " ".join(batch["label"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = train_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_data.column_names)
    vocab_dev = dev_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dev_data.column_names)
    # print(vocab_train["vocab"][0])
    
    #-------------------Create a vocabullary dictionary-------------------------------- 

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_dev["vocab"][0]))
    # sys.exit()
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    print(vocab_dict)

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]


    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(len(vocab_dict))

    
    with open('/home/elias/code/checkpoints/vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    
    tokenizer = Wav2Vec2CTCTokenizer("/home/elias/code/checkpoints/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


    try:
        processor = Wav2Vec2Processor.from_pretrained("/home/elias/code/checkpoints/")
    except:
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        
    
    processor.save_pretrained("/home/elias/code/checkpoints/")

    
    def speech_file_to_array_fn(batch):

        speech_array, sampling_rate = torchaudio.load(batch["audio_path"] )
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["label"]
        
        return batch

    train_data = train_data.map(speech_file_to_array_fn, remove_columns = train_data.column_names)
    dev_data = dev_data.map(speech_file_to_array_fn, remove_columns=dev_data.column_names)
    
    def resample(batch):
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48000, 16000)
        batch["sampling_rate"] = 16000
        return batch

    train_data = train_data.map(resample, num_proc=4)
    dev_data = dev_data.map(resample, num_proc=4)



    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names, batch_size=32, num_proc=4, batched=True)
    dev_data = dev_data.map(prepare_dataset, remove_columns=dev_data.column_names, batch_size=32, num_proc=4, batched=True)


    # --------------------------Training pipeline-------------------------------------------------------# 

    @dataclass
    class DataCollatorCTCWithPadding:
        
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch


    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


    wer_metric = load_metric("wer")


    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    try:
        model = Wav2Vec2ForCTC.from_pretrained("/home/elias/code/checkpoints/").to("cuda")
    except:
        model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    


    model.freeze_feature_extractor()

    training_args = TrainingArguments(
    output_dir= "/home/elias/code/checkpoints/",
    group_by_length = True,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    gradient_accumulation_steps = 4,
    gradient_checkpointing= True,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=4e-4,
    warmup_steps=int(0.1*3600),
    save_total_limit=2,
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=processor.feature_extractor,
    )

    mlflow.set_tracking_uri("SET YOUR URL") 
    experiment_id = mlflow.get_experiment_by_name("stt_Wav2Vec2_base_phoneme_original")
    if experiment_id is None:
        experiment_id = mlflow.create_experiment("stt_Wav2Vec2_base_phoneme_original")
    else:
        experiment_id = experiment_id.experiment_id
    
    
    print(torch.cuda.memory_summary())
    
    with mlflow.start_run(experiment_id= experiment_id):
        trainer.train()   


    # Saving the model's checkpoint

    trainer.save_model('/home/elias/code/checkpoints/')
    
    parameters.update({"last_batch_checkpoint": last_index})
    with open("/home/elias/code/model/properties.json", "w") as outfile:
        json.dump(parameters, outfile)
        
    #Delete the trainer to reduce the cache and others parameters    
    # del trainer
    
    # with no_torch():
    #     trainer.no_grad()
    

if __name__=='__main__':
    print("Here")
    
    list_of_wavs = glob.glob("SET THE DATA PATH")
    print(len(list_of_wavs))
    
    sys.exit()
    
    last_check_point = parameters['last_batch_checkpoint']
    
    batch_path_list=[]
         
    for i in range(last_check_point+1, len(list_of_wavs)):
        batch_path_list.append(list_of_wavs[i])
        
        if len(batch_path_list)==batch_size or i==len(list_of_wavs)-1:
            parameters.update({"running": 1})
            print('Model is in active stage')
            with open("/home/elias/code/model/properties.json", "w") as outfile:
                json.dump(parameters, outfile)
                
                
            start_train(batch_path_list, i)
            
            parameters.update({"running": 0})
            print('Model is in stop stage')
            
            with open("/home/elias/code/model/properties.json", "w") as outfile:
                json.dump(parameters, outfile)
            
            batch_path_list.clear()
            
            print('Batch complete')
            
            with torch.no_grad():
                
                gc.collect()
                torch.cuda.empty_cache()
                
            print(torch.cuda.memory_summary()) 
            
            break
            
    
    
    
    
        
    
    
    
    
    
