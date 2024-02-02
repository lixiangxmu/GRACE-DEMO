
import re
import penman


##############################################################################
def svo_r(parse,stopword_all):
    verbs = [verb for verb in parse if verb.pos_ == "VERB" and verb.lemma_.lower() not in stopword_all and len(verb.lemma_)>=3]
    events = []
    if verbs == []:
        events = [['--null--', '--null--', '--null--']]
    for verb in verbs:  
        event = ['--null--', '--null--', '--null--']
        event[1] = re.sub('[^A-Za-z]+', '', verb.lemma_.lower())
        for argument in verb.children:
            if argument.dep_ in {"subj","nsubj",  "csubj",  "xsubj"}:  
                if argument.lemma_.lower() not in stopword_all and len(argument.lemma_)>=3 and argument.pos_ == "NOUN":
                    event[0] = argument.lemma_.lower() 
                    continue
        for argument in verb.children:
            if argument.dep_ in {"obj","dobj", "iobj", "pobj","lobj","nsubjpass","csubjpass","npsubj"}: 
                if argument.lemma_.lower() not in stopword_all and len(argument.lemma_)>=3 and argument.pos_ == "NOUN":
                    event[2] = argument.lemma_.lower() 
                    break
        events.append(event)
    return events


def extract_event_by_spacy(text,nlp,stopword_all):
    parse = nlp(text)
    eventss = []
    events=svo_r(parse,stopword_all) 
    print(events)
    for event in  events:
        wcount=0
        for w in event:
            if w !='--null--' :
                wcount+=1
        if wcount>=2:
            eventss.append(event)
    return eventss,parse 

##############################################################################

def filter_arm_event(events,text,parse,stopword_spacy):
    new_events=[]
    doc0=parse
    pos0={k.lemma_.lower():k.pos_ for k in doc0}
    for event in events:
        lemma1=event.split(' ')
        pos1=[]
        if lemma1[0] in pos0 and lemma1[1] in pos0 and lemma1[0] not in stopword_spacy and lemma1[1] not in stopword_spacy:
            pos1=[pos0[lemma1[0]],pos0[lemma1[1]]]

        if pos1==['NOUN','VERB'] or pos1==['VERB','NOUN'] :
            new_events.append(' '.join(lemma1))
    return new_events

def get_amrarg_clean(text_ori,stopword_spacy):
        text=text_ori.replace("\"",'').replace("-",' ')
        text = re.sub(r'[0-9]+', '',text)
        text = re.sub('\s+', ' ', text).strip()
        text =text.split(' ')
        if len(text)>=2:
            return ''
        txt=text[0]
        if txt and txt not in stopword_spacy:
            return txt
        else:
            return ''

def extract_event_by_amr(txt,parse,nlp,stopword_spacy):
    g=penman.decode(txt)
    triples=[]
    edges=[edge for edge in g.edges() if 'ARG' in edge.role]
    pair=None
    node_label={}
    for inst in  g.instances():
        node_label[str(inst.source)]=str(inst.target)
    for e in edges:
        a, b,l=node_label[str(e.source)], node_label[str(e.target)], str(e.role)
        a_c, b_c=get_amrarg_clean(a,stopword_spacy),get_amrarg_clean(b,stopword_spacy)
        if 'ARG0' in l:
            pair=b_c+' '+a_c
        else:
            pair=a_c+' '+b_c
        if len(pair.strip().split(' '))==2:
            triples.append(pair)
    print(triples)
    if triples:
        triples=filter_arm_event(triples,g.metadata['snt'],parse,stopword_spacy)
    return triples





