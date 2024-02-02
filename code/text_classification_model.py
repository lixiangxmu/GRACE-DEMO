
from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel,AutoConfig
from torch.utils.data import DataLoader, Dataset


class MyDataSet(Dataset):
    def __init__(self, data,configs):
        self.text = data
        self.max_length=configs['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(configs['model_name'])  

    def __len__(self): 
        return len(self.text)

    def __getitem__(self, index):
        cur_text=self.text[index]
        encoded_input2= self.tokenizer(cur_text,max_length=self.max_length,truncation=True,return_tensors="pt") 
        return encoded_input2['input_ids'],encoded_input2['attention_mask']

def load_data(configs,ori_data):
    info_train_loader=DataLoader(MyDataSet([ori_data],configs), batch_size=configs['batch_size'],shuffle=False)
    return info_train_loader


class MyModel(nn.Module):
    def __init__(self,configs):
        super(MyModel, self).__init__()
        self.configs=configs   
        self.bert_model =  AutoModel.from_pretrained(configs['model_name']) 
        self.bert_model_config = AutoConfig.from_pretrained(configs['model_name']) 
        self.bert_model_hidden_size=self.bert_model_config.hidden_size
        self.fcs_info=nn.Linear(self.bert_model_hidden_size,2,bias=True)

    def forward(self,input_ids,attention_mask):
        input_ids=input_ids.cuda()
        attention_mask=attention_mask.cuda()
        input_ids2=input_ids.reshape(-1,input_ids.shape[-1]) 
        attention_mask2=attention_mask.reshape(-1,input_ids.shape[-1]) 
        bert_output=self.bert_model(input_ids=input_ids2, attention_mask=attention_mask2) 
        bert_pooler_output=bert_output['pooler_output']
        out_bert=bert_pooler_output
        bert_prob=self.fcs_info(out_bert)
        return  bert_prob



