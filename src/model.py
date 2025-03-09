import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv3D, GlobalMaxPooling3D, concatenate

def build_model():
    # Аудио ветвь (13 MFCC + 1 pitch + 1 RMS = 15 признаков)
    audio_input = Input(shape=(15,))
    audio_dense = Dense(32, activation='relu')(audio_input)
    
    # МРТ ветвь (3D-CNN)
    mri_input = Input(shape=(64, 64, 64, 1))
    x = Conv3D(8, (3,3,3), activation='relu')(mri_input)
    x = GlobalMaxPooling3D()(x)
    mri_dense = Dense(32, activation='relu')(x)
    
    # Объединение
    combined = concatenate([audio_dense, mri_dense])
    output = Dense(1, activation='sigmoid')(combined)
    
    model = tf.keras.Model(inputs=[audio_input, mri_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model