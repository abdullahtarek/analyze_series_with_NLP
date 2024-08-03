from transformers import pipeline
import torch 
from glob import glob
import pandas as pd
import numpy as np
import os
from nltk import sent_tokenize
import nltk
import sys 
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import load_subtitles_dataset
nltk.download('punkt')

class ThemeClassifier():
    def __init__(self,theme_list):
        self.model_name = 'facebook/bart-large-mnli'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_classifier = self.load_model(self.device)
        self.theme_list = theme_list #['friendship','hope', 'sacrifice', 'battle','self development','betrayal','love','dialogue']
        #friendship,hope,sacrifice,battle,self development,betrayal,love,dialogue


    def load_model(self,device):
        theme_classifier = pipeline("zero-shot-classification",
                      model= self.model_name,
                      device=device)
        return theme_classifier

    def get_themes_inference(self,script):
        script_sentences = sent_tokenize(script)

        # Batch sentences 
        script_batches = []
        sentence_batch_size=20
        for index in range(0,len(script_sentences),sentence_batch_size):
            sent = " ".join(script_sentences[index:index+sentence_batch_size])
            script_batches.append(sent)
        
        emotion_output = self.theme_classifier(script_batches, 
                        self.theme_list,multi_label=True)

        # Wrangle Output    
        emotions={}
        for output in  emotion_output:
            for label, score in zip(output['labels'],output['scores']):
                if label not in emotions:
                    emotions[label]=[]
                emotions[label].append(score)
        
        emotions =  {key:np.mean(np.array(value)) for key,value in emotions.items()}
    
        return emotions

    def get_themes(self,dataset_path,save_path=None):
        # Read Saved output if exists
        if save_path is not None and os.path.exists(save_path):
            return pd.read_csv(save_path)

        # Load Dataset
        df = load_subtitles_dataset(dataset_path)

        # Run Inference
        output_emotions = df['script'].apply(self.get_themes_inference)

        # Wrangle Output to add the output in a dataframe
        emotion_df = pd.DataFrame(output_emotions.tolist())
        df[emotion_df.columns] = emotion_df
        
        # Save Output
        if save_path is not None:
            df.to_csv(save_path,index=False)

        return df