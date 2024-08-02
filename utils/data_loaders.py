from glob import glob 
import pandas as pd

def load_subtitles_dataset(dataset_path):
    substitles_paths = sorted(glob(dataset_path+"/*.ass"))
    scripts = []
    episode_num = []
    for path in substitles_paths:
        
        # Read Lines
        with open(path,'r') as file:
            lines = file.readlines()
            lines = lines[27:]
            rows = [",".join(line.split(',')[9:]) for line in lines]
        
        # Clean Output
        rows = [line.replace("\\N",' ') for line in rows]
        script = " ".join(rows)
        
        # Get Episode Number
        episode = int(path.split('-')[1].split('.')[0].strip())
        
        scripts.append(script)
        episode_num.append(episode)
    
    df = pd.DataFrame.from_dict({'episode':episode_num,'script':scripts})
    return df