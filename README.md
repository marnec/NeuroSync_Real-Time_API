# NeuroSync_Real-Time_API
A collection of AI model endpoints you can run locally for a real-time audio2face system.

You will need to download the models from huggingface and put them in the correct folders (kokoro installs automatically, dont worry about that one - just do the correct pip install first : pip install -q kokoro>=0.8.2 soundfile )

Take note of licences, for your use case. 

Text to Embeddings : https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5 

NeuroSync Audio2Face : https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape 

Whisper_Turbo (STT) : https://huggingface.co/openai/whisper-large-v3-turbo

Vision (image to text) : https://huggingface.co/Salesforce/blip-image-captioning-large  |  https://huggingface.co/openai/clip-vit-large-patch14  |  https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
