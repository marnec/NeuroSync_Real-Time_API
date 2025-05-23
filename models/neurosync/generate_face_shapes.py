# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# generate_face_shapes.py

# import numpy as np

from models.neurosync.audio.extraction.extract_features import extract_audio_features
from models.neurosync.audio.processing.audio_processing import process_audio_features

def generate_facial_data_from_bytes(audio_bytes, model, device, config):
    
    audio_features, y = extract_audio_features(audio_bytes, from_bytes=True)
    
  #  if audio_features is None or y is None:
    #    return [], np.array([])
  
    final_decoded_outputs = process_audio_features(audio_features, model, device, config)

    return final_decoded_outputs

