import nibabel as nib
import numpy as np
from skimage.transform import resize

def preprocess_mri(file_path: str, target_shape=(64, 64, 64)) -> np.ndarray:
    # Преобразование МРТ-скана в 3D-массив
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        data_resized = resize(data_norm, target_shape, anti_aliasing=True)
        return np.expand_dims(data_resized, axis=-1)  # Добавляем размерность канала
    except Exception as e:
        print(f"Ошибка в {file_path}: {str(e)}")
        return None