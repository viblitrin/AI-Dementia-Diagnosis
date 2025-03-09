import librosa
import numpy as np

def extract_audio_features(file_path: str, sr: int = 16000) -> dict:
    # Извлечение признаков из аудио (MFCC, тон, RMS)
    try:
        signal, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        pitch = librosa.yin(signal, fmin=50, fmax=500)
        rms = librosa.feature.rms(y=signal)
        
        return {
            "mfcc": np.mean(mfcc, axis=1).tolist(),
            "pitch": np.nanmean(pitch).item(),
            "rms": np.mean(rms).item()
        }
    except Exception as e:
        print(f"Ошибка в {file_path}: {str(e)}")
        return None