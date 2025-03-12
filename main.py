import os
from threading import Thread
from flask import Flask, request, jsonify, Response
from datetime import datetime
import json
import traceback
import time  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose which GPU to use, 0 is main.

import torch

from neurosync.config import config

from utils_emb import process_embedding
from utils_audio import process_transcription, process_blendshapes, generate_speech_segment_tts 
from utils_image import process_image, process_clip, process_pdf_imagery

from model_loader import load_audio_models, load_embedding_model, load_image_models, load_tts_model

LOG_FILE_PATH = "log"
os.makedirs(LOG_FILE_PATH, exist_ok=True)

def log_event(event_type, status, details=None):
    """Logs events such as transcriptions, embeddings, and blendshape generations."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "status": status,
        "details": details
    }
    log_file = os.path.join(LOG_FILE_PATH, "app_log.jsonl")
    with open(log_file, "a") as file:
        file.write(json.dumps(log_entry) + "\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype_audio = torch.bfloat16 if torch.cuda.is_available() else torch.float32
torch_dtype_image = torch.float16 if torch.cuda.is_available() else torch.float32

# ============================================================
# Load Models Using model_loader.py
# ============================================================
audio_models = load_audio_models()
transgenerator = audio_models["transgenerator"]
blendshape_model = audio_models["blendshape_model"]

# Load the embedding model using its dedicated function
embmodel = load_embedding_model()

image_models = load_image_models()
blip_processor = image_models["blip_processor"]
blip_model = image_models["blip_model"]
clip_processor = image_models["clip_processor"]
clip_model = image_models["clip_model"]
vilt_processor = image_models["vilt_processor"]
vilt_model = image_models["vilt_model"]

# NEW: Load the TTS model (e.g. Kokoro) using its dedicated function
tts_models = load_tts_model()
# tts_models is a dict with keys "tts_pipeline" and "tts_lock"

# ------------------- Audio API (Port 6969) -------------------
app_audio = Flask("audio_app")

@app_audio.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        audio_base64 = request.json.get('audio_base64')
        if not audio_base64:
            log_event("transcription", "failure", "No audio data provided.")
            return jsonify({"status": "error", "message": "No audio data provided."}), 400

        return_timestamps = request.json.get('return_timestamps', False)
        result = process_transcription(audio_base64, return_timestamps, transgenerator)
        log_event("transcription", "success", result)
        return jsonify(result)
    except Exception as e:
        err_msg = str(e)
        log_event("transcription", "failure", err_msg)
        return jsonify({"status": "error", "message": err_msg}), 500

@app_audio.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    try:
        audio_bytes = request.data
        if not audio_bytes:
            msg = "No audio data provided."
            log_event("blendshapes", "failure", msg)
            return jsonify({"status": "error", "message": msg}), 400
        result = process_blendshapes(audio_bytes, blendshape_model, device, config)
        log_event("blendshapes", "success", "Blendshapes generated successfully.")
        return jsonify(result)
    except Exception as e:
        err_msg = str(e)
        print("Exception occurred:", err_msg)
        print(traceback.format_exc())
        log_event("blendshapes", "failure", err_msg)
        return jsonify({"status": "error", "message": err_msg}), 500

@app_audio.route('/get_embedding', methods=['POST'])
def get_embedding():
    try:
        data = request.json
        if not data or 'text' not in data:
            log_event("embedding", "failure", "No text data provided.")
            return jsonify({"status": "error", "message": "No text data provided."}), 400
        text = data['text']
        result = process_embedding(text, embmodel)
        log_event("embedding", "success", f"Generated embedding for text: {text}")
        return jsonify(result)
    except Exception as e:
        err_msg = str(e)
        log_event("embedding", "failure", err_msg)
        return jsonify({"status": "error", "message": err_msg}), 500

# ------------------- Image API (Port 1234) -------------------
app_image = Flask("image_app")

@app_image.route("/process_image", methods=["POST"])
def route_process_image():
    try:
        data = request.data.decode("utf-8")
        result = process_image(data, blip_processor, blip_model, clip_processor, clip_model, vilt_processor, vilt_model, device)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app_image.route("/process_clip", methods=["POST"])
def route_process_clip():
    try:
        data = request.data.decode("utf-8")
        result = process_clip(data, clip_processor, clip_model, device)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app_image.route("/process_pdf_imagery", methods=["POST"])
def route_process_pdf_imagery():
    try:
        data = request.data.decode("utf-8")
        result = process_pdf_imagery(data, blip_processor, blip_model, clip_processor, clip_model, vilt_processor, vilt_model, device)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------------- TTS API (Port 8000) -------------------
app_tts = Flask("tts_app")

@app_tts.route('/generate_speech', methods=['POST'])
def generate_speech_tts_endpoint():
    """
    Endpoint for generating speech using the TTS engine.
    """
    text = request.json.get('text', '')
    # UPDATED: Pass the loaded TTS pipeline and lock to the utility function
    result = generate_speech_segment_tts(text, tts_models["tts_pipeline"], tts_models["tts_lock"])
    if result is None:
        print("TTS engine failed to generate audio.")
        return jsonify({"error": "Failed to generate audio with TTS engine"}), 500
    else:
        return result, 200, {'Content-Type': 'audio/wav'}
    



@app_tts.route("/synthesize_and_blendshapes", methods=["POST"])
def synthesize_and_blendshapes():
    start_time = time.time()  # Record start time
    try:
        data = request.json
        text = data.get("text", "").strip()
        if not text:
            log_event("synthesize_and_blendshapes", "failure", "No text data provided.")
            return jsonify({"status": "error", "message": "No text data provided."}), 400

        # Pass TTS model parameters to generate_speech_segment_tts
        audio_bytes = generate_speech_segment_tts(text, tts_models["tts_pipeline"], tts_models["tts_lock"])
        if audio_bytes is None:
            log_event("synthesize_and_blendshapes", "failure", "Failed to generate speech with TTS engine.")
            return jsonify({"status": "error", "message": "Failed to generate speech with TTS engine."}), 500

        blendshapes_result = process_blendshapes(audio_bytes, blendshape_model, device, config)
        blendshapes_list = blendshapes_result.get("blendshapes", [])
        boundary = "MY_BOUNDARY_STRING"

        part1 = (
            f"--{boundary}\r\n"
            "Content-Type: audio/wav\r\n"
            "Content-Disposition: attachment; filename=\"output.wav\"\r\n"
            "\r\n"
        ).encode('utf-8') + audio_bytes + "\r\n".encode('utf-8')
        part2_json = json.dumps(blendshapes_list).encode('utf-8')
        part2 = (
            f"--{boundary}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Disposition: inline\r\n"
            "\r\n"
        ).encode('utf-8') + part2_json + "\r\n".encode('utf-8')

        closing_boundary = f"--{boundary}--\r\n".encode('utf-8')
        multipart_body = part1 + part2 + closing_boundary

        response = Response(multipart_body, status=200)
        response.headers["Content-Type"] = f"multipart/mixed; boundary={boundary}"
        log_event("synthesize_and_blendshapes", "success", "Speech and blendshapes generated successfully.")

        # Calculate and print elapsed time
        elapsed_time = time.time() - start_time
        print(f"Time taken for synthesize_and_blendshapes: {elapsed_time:.2f} seconds")
        
        return response
    except Exception as e:
        err_msg = str(e)
        log_event("synthesize_and_blendshapes", "failure", err_msg)
        return jsonify({"status": "error", "message": err_msg}), 500


# ------------------- Embedding API (Port 7070) -------------------
app_embedding = Flask("embedding_app")

@app_embedding.route('/get_embedding', methods=['POST'])
def get_embedding_embedding():
    try:
        data = request.json
        if not data or 'text' not in data:
            log_event("embedding", "failure", "No text data provided.")
            return jsonify({"status": "error", "message": "No text data provided."}), 400
        text = data['text']
        result = process_embedding(text, embmodel)
        log_event("embedding", "success", f"Generated embedding for text: {text}")
        return jsonify(result)
    except Exception as e:
        err_msg = str(e)
        log_event("embedding", "failure", err_msg)
        return jsonify({"status": "error", "message": err_msg}), 500

# ------------------- Run All Flask Apps -------------------
if __name__ == '__main__':
    def run_audio_app():
        print("Starting Audio App on port 6969...")
        app_audio.run(host='0.0.0.0', port=6969, debug=False)

    def run_image_app():
        print("Starting Image App on port 1234...")
        app_image.run(host='0.0.0.0', port=1234, debug=False)

    def run_tts_app():
        print("Starting TTS App on port 8000...")
        app_tts.run(host='0.0.0.0', port=8000, debug=False)

    def run_embedding_app():
        print("Starting Embedding App on port 7070...")
        app_embedding.run(host='0.0.0.0', port=7070, debug=False)

    t_audio = Thread(target=run_audio_app)
    t_image = Thread(target=run_image_app)
    t_tts = Thread(target=run_tts_app)
    t_embedding = Thread(target=run_embedding_app)

    t_audio.start()
    t_image.start()
    t_tts.start()
    t_embedding.start()

    t_audio.join()
    t_image.join()
    t_tts.join()
    t_embedding.join()
