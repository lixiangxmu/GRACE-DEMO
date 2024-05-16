
import warnings
import streamlit as st
import streamlit.components.v1 as components
import re
from text_classification_model import *
from event_extract import *
import amrlib
import torch
import spacy
import json
import torch.nn.functional as F
import transformers
import gpt_2_simple as gpt2


######################################################################################
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

######################################################################################

st.set_page_config(page_title="GRACE", initial_sidebar_state="collapsed", page_icon='✨', layout="centered") #auto, page_icon=":taxi:"   wide centered

######################################################################################
def clean_text(text):
    tweet_words = text.split(' ')
    tWords = []
    for word in tweet_words:
        word = word.strip()
        if ((len(word) > 1 and word[0] == '@')):
            continue
        elif (word.lower() == "rt" or word.lower() == "userid"):
            continue
        elif (len(word) >= 1):
            tWords.append(word)
    text = " ".join(tWords)
    text = re.sub(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', '', text) #删除网址链接
    text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', '', text)  #删除日期
    text = re.sub(r'\d{1,2}:\d{1,2}:\d{1,2}', '', text)
    text = re.sub('\s+', ' ', text).strip()
    return text
######################################################################################

def load_spacy(model_name):
    nlp = spacy.load(model_name)
    stopword_spacy = nlp.Defaults.stop_words
    return nlp,stopword_spacy
nlp,stopword_spacy=load_spacy("en_core_web_trf")


######################################################################################

def extract_by_amr(input_text,amr_model,parse,nlp,stopword_spacy):
    graphs = amr_model.parse_sents([input_text])	
    res=extract_event_by_amr(graphs[0],parse,nlp,stopword_spacy)
    return res

def extract_by_spacy(text,nlp,stopword_spacy):
    events,parse=extract_event_by_spacy(text,nlp,stopword_spacy)
    new_event=[]
    for event in events:
        if '--null--' not in event[:2]:
            new_event.append(' '.join(event[:2]))
        if '--null--' not in event[1:]:
            new_event.append(' '.join(event[1:]))
    return new_event,parse

# ######################################################################################

@st.cache(allow_output_mutation=True)
def load_amr_model():
    model_large='/home/lzy/HDD1/anaconda3/envs/jxxdm/lib/python3.7/site-packages/amrlib/data/model_stog_large/'
    stog = amrlib.load_stog_model(model_large)
    return stog 
amr_model = load_amr_model()

######################################################################################

@st.cache(allow_output_mutation=True)
def get_gpt(PREFIX_ALL):
    BATCH_SIZE=10
    cur_run_name='run18-124M'
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess,run_name=cur_run_name,checkpoint_dir='/home/lzy/HDD1/bl/xgulu/backup/crisis_transl_graph/data/causalbank/checkpoint')
    result_all=[]
    cause_flag='because'
    effect_flag='thus'
    for PREFIX in PREFIX_ALL:
        PREFIX1=PREFIX+' '+cause_flag,
        PREFIX2=PREFIX+' '+effect_flag,
        PREFIXS=[PREFIX1[0],PREFIX2[0]]
        results=[]
        for PREFIX in PREFIXS:
            text =gpt2.generate(sess,
                            run_name=cur_run_name,
                            checkpoint_dir='/home/lzy/HDD1/bl/xgulu/backup/crisis_transl_graph/data/causalbank/checkpoint',
                            length=2,
                            temperature=0.8,
                            prefix=PREFIX,
                            nsamples=20,
                            batch_size=BATCH_SIZE,
                            top_k=20, 
                            return_as_list=True,
                            seed=22,
                            ) 
            new_text=[]
            for txt in text:
                new_text.append(txt.replace(PREFIX+' ',''))
            results.append(new_text)
        result_all.append(results)
    return result_all

######################################################################################

configs = dict(
        model_name='/home/lzy/HDD1/bl/xgulu/backup/pre_lm/roberta-large',
        max_length=100,
        batch_size =1,
    )

@st.cache(allow_output_mutation=True)
def load_info_model():
    info_model= MyModel(configs)
    info_model.load_state_dict(torch.load('/home/lzy/HDD1/bl/xgulu/backup/crisis_transl_graph/data/info_identify/code/model_save/final/model_0.892.h5'))
    return info_model.cuda()
info_model = load_info_model()


def evaluate(model_info, loader):
    model_info.eval()
    y_pred = []
    for i,(input_ids,attention_mask) in enumerate(loader):
        with torch.cuda.amp.autocast():
            outputs= model_info(input_ids,attention_mask)
        y_pred_max=torch.max(outputs, 1)[1].tolist()
        y_pred=F.softmax(outputs,dim=-1).tolist()
        y_pred=[round(x, 1) for x in y_pred[0]]
    return y_pred,y_pred_max[0]

@st.cache(allow_output_mutation=True)
def pred_info(text):
    data_loader=load_data(configs,text)
    pred_info,pred_info_max=evaluate(info_model, data_loader)
    return pred_info,pred_info_max


######################################################################################


def pipeline():
    

    st.markdown("### 🏭 GRACE")
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css") #streamlit.css style.css
    
    row1_1, row1_2= st.columns((0.85, 0.15))
    with row1_1:
        content_input=st.text_input("Paste your text below","The heavy snow block all the road.", key='input',
        help='Example texts:\
        \n(1) Tech companies helping out hurricane-damaged Puerto Rico.\
        \n(2) The building could collapse at any moment.') ##只要输入文本，就赋值给content,height=200,max_chars=140,
        content_clean=clean_text(content_input)
    with row1_2:
        st.write('')
        st.write('')
        st.button(label="✨  start!")

    info_value,info_value_max=pred_info(content_clean)
    whether_info='Related' if info_value_max==1 else 'Unrelated'

    components.html("""<hr style="height:0.5rem;border:none;color:#48a140ba;background-color:#48a140ba;" /> """,height=10)
        
    event_by_spacy,parse=extract_by_spacy(content_clean,nlp,stopword_spacy)
    event_by_amr=extract_by_amr(content_clean,amr_model,parse,nlp,stopword_spacy)

    event_list=list(set(event_by_spacy+event_by_amr))

    row3_1, row3_2= st.columns((1.2, 2))
    if event_list:
        result_all=get_gpt(event_list)
        event_list_map={k:i for i,k in enumerate(event_list)}
        with row3_1:
            with st.expander("It is "+whether_info.lower()+'!',expanded=True):
                cgraph = open("show_text_classification.html", encoding="utf-8")
                components.html(cgraph.read().replace('unrelated_value',str(info_value[0])) .replace('related_value',str(info_value[1]))
                                ,height=90) #,scrolling=True
            with st.expander("Select a sub-event!",expanded=True):
                select_event=st.radio("", event_list)

            results=result_all[event_list_map[select_event]]
            print(results)

        with row3_2:        
            topk_num=5
            result_cause=[]
            result_effect=[]
            for e in results[0]:
                e=e.replace(' ','\n')
                if e not in result_cause and e!=select_event.replace(' ','\n') and len(e.split('\n'))==2:
                    result_cause.append(e)
                    if len(result_cause)==topk_num:
                        break
            for e in results[1]:
                e=e.replace(' ','\n')
                if e not in result_effect and e!=select_event.replace(' ','\n') and e not in result_cause and len(e.split('\n'))==2:
                    result_effect.append(e)
                    if len(result_effect)==topk_num:
                        break

            causal_json={
                        "data": [
                        {"id":0 ,"name":select_event.replace(' ','\n'),"category":0 },
                        {"id":1 ,"name":result_cause[0],"category":1 },
                        {"id":2 ,"name":result_cause[1],"category":1 },
                        {"id":3 ,"name":result_cause[2],"category":1 },
                        {"id":4 ,"name":result_cause[3],"category":1},
                        {"id":5 ,"name":result_cause[4],"category":1 },
                        {"id":6 ,"name":result_effect[0],"category":2 },
                        {"id":7 ,"name":result_effect[1],"category":2},
                        {"id":8 ,"name":result_effect[2],"category":2 },
                        {"id":9 ,"name":result_effect[3],"category":2},
                        {"id":10 ,"name":result_effect[4],"category":2 }], 
                        "links": [
                        {"source": 0, "target": 1, "value": ""}, 
                        {"source": 0, "target": 2, "value": ""}, 
                        {"source": 0, "target": 3, "value": ""},
                        {"source": 0, "target": 4, "value": ""},
                        {"source": 0, "target": 5, "value": ""},
                        {"source": 6, "target": 0, "value": ""}, 
                        {"source": 7, "target": 0, "value": ""}, 
                        {"source": 8, "target": 0, "value": ""},
                        {"source": 9, "target": 0, "value": ""},
                        {"source": 10, "target": 0, "value": ""},
                    ]}
            node_size=3*15
            cgraph = open("shwo_causal_graph.html", encoding="utf-8")
            components.html(cgraph.read().replace('causal_json',json.dumps(causal_json)).replace('node_size',str(node_size))
                                ,height=280) #,scrolling=True
    else:
        with st.expander("It is "+whether_info.lower()+'!',expanded=True):
            cgraph = open("show_text_classification.html", encoding="utf-8")
            print('---',info_value)
            components.html(cgraph.read().replace('unrelated_value',str(info_value[0])).replace('related_value',str(info_value[1])) 
                            ,height=90) #,scrolling=True
        

if __name__ == '__main__':

    pipeline()

