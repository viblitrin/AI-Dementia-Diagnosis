import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def save_metadata(features, labels, output_path):
    # Сохранение метаданных с обработкой пропущенных значений
    df = pd.DataFrame(features)
    df['label'] = labels
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)
    
def load_and_split_data(audio_path, mri_path, test_size=0.2):
    # Загрузка и разделение данных
    audio_data = np.load(audio_path)
    mri_data = np.load(mri_path)
    labels = pd.read_csv('data/labels.csv')['label'].values
    
    return train_test_split(
        [audio_data, mri_data], 
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )