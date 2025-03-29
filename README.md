# NeuroSync_Real-Time_API

## 29/03/2025 Update to NeuroSync model.pth and model.py

- Increased accuracy (timing and overall face shows more natural movement overall, brows, squint, cheeks + mouth shapes)
- More smoothness during playback (flappy mouth be gone in most cases, even when speaking quickly)
- Works better with more voices and styles of speaking.
- This preview of the new model is a modest increase in capability that requires both model.pth and model.py to be replaced with the new versions.

[Download the model from Hugging Face](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape)

# A collection of AI model endpoints you can run locally for a real-time audio2face system.

*Used with [The Player](https://github.com/AnimaVR/NeuroSync_Player), NeuroSync real-time api creates facial blendshape animations that are sent to Unreal Engine via LiveLink from audio or text inputs, locally.*

It also provides extra endpoints for modular AI model integrations - expand your AI's capabilities with embeddings, image to text vision models and audio to text from whisper.

You will need to download the models from huggingface and put them in the correct folders (kokoro installs automatically, dont worry about that one - just do the correct pip install first : pip install -q kokoro>=0.8.2 soundfile )

Take note of licences, for your use case. 

Text to Embeddings : https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5 

NeuroSync Audio2Face : https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape 

Whisper_Turbo (STT) : https://huggingface.co/openai/whisper-large-v3-turbo

Vision (image to text) : https://huggingface.co/Salesforce/blip-image-captioning-large  |  https://huggingface.co/openai/clip-vit-large-patch14  |  https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
