import os
import sys
import runpod
import tempfile
import base64
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add HebTTSLM to path
sys.path.insert(0, str(Path(__file__).parent / "HebTTSLM"))

from HebTTSLM.infer import prepare_inference, infer_texts
from HebTTSLM.utils import AttributeDict
from chunked_inference import infer_chunked_text
from text_chunker import HebrewTextChunker

# Global variables for model (loaded once at startup)
model_components = None
speakers_config = None

def load_model():
    """Load the HebTTS model once at container startup"""
    global model_components, speakers_config
    
    print("Loading HebTTS model...")
    
    hebtts_dir = Path(__file__).parent / "HebTTSLM"
    checkpoint_path = hebtts_dir / "checkpoint.pt"
    speakers_yaml_path = hebtts_dir / "speakers" / "speakers.yaml"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load speakers config
    speakers_config = OmegaConf.load(speakers_yaml_path)
    
    # Create args
    args = AttributeDict({
        'tokens_file': str(hebtts_dir / "tokenizer" / "unique_words_tokens_all.k2symbols"),
        'vocab_file': str(hebtts_dir / "tokenizer" / "vocab.txt"),
        'mbd': True,
        'text_tokens_path': str(hebtts_dir / "tokenizer" / "unique_words_tokens_all.k2symbols")
    })
    
    # Use default speaker for initial model load
    default_speaker = "osim"
    speaker_info = speakers_config[default_speaker]
    audio_prompt_path = hebtts_dir / "speakers" / speaker_info["audio-prompt"]
    
    # Prepare inference
    device, model, text_collater, audio_tokenizer, alef_bert_tokenizer, audio_prompts = prepare_inference(
        str(checkpoint_path), args, str(audio_prompt_path)
    )
    
    model_components = {
        'device': device,
        'model': model,
        'text_collater': text_collater,
        'audio_tokenizer': audio_tokenizer,
        'alef_bert_tokenizer': alef_bert_tokenizer,
        'audio_prompts': audio_prompts,
        'args': args
    }
    
    print("Model loaded successfully!")

def handler(job):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "text": "Hebrew text here",
            "speaker": "osim",
            "top_k": 50,
            "temperature": 1,
            "use_mbd": true,
            "filename": "output",
            "enable_chunking": true,
            "max_chunk_chars": 150
        }
    }
    """
    try:
        job_input = job["input"]
        
        # Validate required fields
        if "text" not in job_input or "speaker" not in job_input:
            return {"error": "Missing required fields: text and speaker"}
        
        text = job_input["text"]
        speaker = job_input["speaker"]
        
        # Optional parameters with defaults
        top_k = job_input.get("top_k", 15)
        temperature = job_input.get("temperature", 0.6)
        use_mbd = job_input.get("use_mbd", True)
        filename = job_input.get("filename", "output")
        enable_chunking = job_input.get("enable_chunking", True)
        max_chunk_chars = job_input.get("max_chunk_chars", 150)
        
        # Validate inputs
        if not text.strip():
            return {"error": "Text cannot be empty"}
        
        if speaker not in speakers_config:
            return {
                "error": f"Invalid speaker: {speaker}",
                "available_speakers": list(speakers_config.keys())
            }
        
        # Get speaker info
        speaker_info = speakers_config[speaker]
        prompt_text = speaker_info["text-prompt"]
        
        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create custom args with user's MBD preference
            custom_args = AttributeDict(model_components['args'])
            custom_args.mbd = use_mbd
            
            # Check if chunking is needed and enabled
            chunker = HebrewTextChunker(max_chunk_chars)
            should_chunk = enable_chunking and not chunker.is_chunk_valid(text)
            
            if should_chunk:
                print(f"Text is {len(text)} characters, using chunking with max {max_chunk_chars} chars per chunk")
                
                # Use chunked inference
                success, audio_file_path, chunk_info = infer_chunked_text(
                    text=text,
                    output_dir=str(output_dir),
                    prompt_text=prompt_text,
                    device=model_components['device'],
                    model=model_components['model'],
                    text_collater=model_components['text_collater'],
                    audio_tokenizer=model_components['audio_tokenizer'],
                    alef_bert_tokenizer=model_components['alef_bert_tokenizer'],
                    audio_prompts=model_components['audio_prompts'],
                    top_k=top_k,
                    temperature=temperature,
                    args=custom_args,
                    base_filename=filename,
                    max_chars=max_chunk_chars
                )
                
                if not success or not audio_file_path or not audio_file_path.exists():
                    return {
                        "error": "Chunked audio generation failed",
                        "chunk_info": chunk_info
                    }
                
                # Read generated audio
                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()
                
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Return audio with chunking info
                return {
                    "audio_base64": audio_base64,
                    "filename": f"{filename}.wav",
                    "sample_rate": 24000,
                    "format": "wav",
                    "chunked": True,
                    "chunk_info": chunk_info,
                    "original_length": len(text),
                    "chunks_processed": len(chunk_info)
                }
            else:
                print(f"Text is {len(text)} characters, processing as single chunk")
                
                # Use standard inference for short texts
                texts_with_filenames = [(filename, text)]
                
                infer_texts(
                    texts_with_filenames=texts_with_filenames,
                    output_dir=str(output_dir),
                    prompt_text=prompt_text,
                    device=model_components['device'],
                    model=model_components['model'],
                    text_collater=model_components['text_collater'],
                    audio_tokenizer=model_components['audio_tokenizer'],
                    alef_bert_tokenizer=model_components['alef_bert_tokenizer'],
                    audio_prompts=model_components['audio_prompts'],
                    top_k=top_k,
                    temperature=temperature,
                    args=custom_args
                )
                
                # Read generated audio
                audio_file_path = output_dir / f"{filename}.wav"
                
                if not audio_file_path.exists():
                    return {"error": "Audio generation failed"}
                
                # Read audio file as base64
                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()
                
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Return audio as base64 in output
                return {
                    "audio_base64": audio_base64,
                    "filename": f"{filename}.wav",
                    "sample_rate": 24000,
                    "format": "wav",
                    "chunked": False,
                    "original_length": len(text)
                }
            
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Processing failed: {str(e)}"}

# Load model at startup (before any jobs)
print("Starting model load...")
load_model()
print("Model load complete, starting serverless worker...")

# Start the serverless worker
runpod.serverless.start({"handler": handler})