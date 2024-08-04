import spacy    
from nltk import sent_tokenize
import pandas as pd
import os
import sys 
import pathlib
from ast import literal_eval
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import load_subtitles_dataset


class NamedEntityRecognizer():
    def __init__(self):
        self.nlp_model = self.load_model()

    def load_model(self):
        nlp_model = spacy.load("en_core_web_trf")
        return nlp_model

    def get_ners_inference(self,script):
        script_sentences = sent_tokenize(script)

        ner_output = []
        
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for ent in doc.ents: 
                if ent.label_=='PERSON':
                    full_name = ent.text
                    first_name=full_name.split(' ')[0]
                    ners.add(first_name)
            ner_output.append(list(ners))
        return ner_output

    def get_ners(self,dataset_path,save_path=None):
        # Read Saved output if exists
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df = df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # Load Dataset
        df = load_subtitles_dataset(dataset_path)  
             
        # Run Inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        # Save Output
        if save_path is not None:
            df.to_csv(save_path,index=False)
        
        return df