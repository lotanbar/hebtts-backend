"""
Chunked Inference Module for Hebrew TTS

This module extends the base inference functionality to handle long texts
through intelligent chunking and audio concatenation.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import torchaudio
import torch

# Import existing components
from text_chunker import prepare_chunked_texts
from HebTTSLM.infer import infer_texts


def concatenate_audio_files(audio_files: List[Path], output_path: Path, sample_rate: int = 24000) -> bool:
    """
    Concatenate multiple audio files into a single output file.
    
    Args:
        audio_files: List of audio file paths to concatenate
        output_path: Output path for concatenated audio
        sample_rate: Sample rate for audio processing
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        audio_tensors = []
        
        for audio_file in audio_files:
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return False
            
            # Load audio file
            waveform, sr = torchaudio.load(audio_file)
            
            # Resample if necessary
            if sr != sample_rate:
                logger.warning(f"Resampling {audio_file} from {sr} to {sample_rate}")
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            audio_tensors.append(waveform)
            logger.debug(f"Loaded {audio_file}: {waveform.shape}")
        
        if not audio_tensors:
            logger.error("No audio tensors to concatenate")
            return False
        
        # Concatenate along time dimension
        concatenated = torch.cat(audio_tensors, dim=1)
        
        # Save concatenated audio
        torchaudio.save(output_path, concatenated, sample_rate)
        logger.info(f"Concatenated {len(audio_files)} files to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error concatenating audio files: {e}")
        return False


def infer_chunked_text(
    text: str,
    output_dir: str,
    prompt_text: str,
    device,
    model,
    text_collater,
    audio_tokenizer,
    alef_bert_tokenizer,
    audio_prompts,
    top_k: int = 50,
    temperature: float = 1.0,
    args=None,
    base_filename: str = "output",
    max_chars: int = 150,
    add_silence_between_chunks: bool = True
) -> Tuple[bool, Optional[Path], List[str]]:
    """
    Process long text through chunking and TTS inference.
    
    Args:
        text: Input text to synthesize
        output_dir: Directory for output files
        prompt_text: Speaker prompt text
        device: PyTorch device
        model: TTS model
        text_collater: Text collation function
        audio_tokenizer: Audio tokenizer
        alef_bert_tokenizer: Hebrew tokenizer
        audio_prompts: Audio prompt tensors
        top_k: Top-k sampling parameter
        temperature: Temperature for sampling
        args: Additional arguments
        base_filename: Base filename for outputs
        max_chars: Maximum characters per chunk
        add_silence_between_chunks: Whether to add brief silence between chunks
        
    Returns:
        Tuple of (success, final_audio_path, chunk_info_list)
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare chunked texts
        texts_with_filenames, is_chunked = prepare_chunked_texts(
            text, base_filename, max_chars
        )
        
        logger.info(f"Processing {'chunked' if is_chunked else 'single'} text: {len(texts_with_filenames)} parts")
        
        # Process each chunk
        chunk_audio_files = []
        chunk_info = []
        
        for i, (filename, chunk_text) in enumerate(texts_with_filenames):
            logger.info(f"Processing chunk {i+1}/{len(texts_with_filenames)}: {filename}")
            logger.debug(f"Chunk text ({len(chunk_text)} chars): {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
            
            # Process single chunk
            infer_texts(
                texts_with_filenames=[(filename, chunk_text)],
                output_dir=str(output_path),
                prompt_text=prompt_text,
                device=device,
                model=model,
                text_collater=text_collater,
                audio_tokenizer=audio_tokenizer,
                alef_bert_tokenizer=alef_bert_tokenizer,
                audio_prompts=audio_prompts,
                top_k=top_k,
                temperature=temperature,
                args=args
            )
            
            # Check if audio file was created
            chunk_audio_path = output_path / f"{filename}.wav"
            if chunk_audio_path.exists():
                chunk_audio_files.append(chunk_audio_path)
                chunk_info.append(f"{filename}: {len(chunk_text)} chars")
                logger.info(f"Generated audio for chunk {i+1}: {chunk_audio_path}")
            else:
                logger.error(f"Failed to generate audio for chunk {i+1}: {filename}")
                return False, None, chunk_info
        
        # If only one chunk, return it directly
        if len(chunk_audio_files) == 1:
            final_path = output_path / f"{base_filename}.wav"
            if chunk_audio_files[0] != final_path:
                # Rename single chunk to final filename
                chunk_audio_files[0].rename(final_path)
            return True, final_path, chunk_info
        
        # Multiple chunks - concatenate them
        if len(chunk_audio_files) > 1:
            logger.info(f"Concatenating {len(chunk_audio_files)} audio files")
            
            if add_silence_between_chunks:
                # Add brief silence between chunks for more natural speech
                logger.debug("Adding silence between chunks")
                extended_audio_files = []
                silence_duration = 0.3  # 300ms silence
                sample_rate = 24000
                
                for i, audio_file in enumerate(chunk_audio_files):
                    extended_audio_files.append(audio_file)
                    
                    # Add silence between chunks (but not after the last one)
                    if i < len(chunk_audio_files) - 1:
                        # Create temporary silence file
                        silence_samples = int(silence_duration * sample_rate)
                        silence_tensor = torch.zeros(1, silence_samples)
                        silence_path = output_path / f"silence_{i}.wav"
                        torchaudio.save(silence_path, silence_tensor, sample_rate)
                        extended_audio_files.append(silence_path)
                
                chunk_audio_files = extended_audio_files
            
            # Concatenate all audio files
            final_audio_path = output_path / f"{base_filename}.wav"
            success = concatenate_audio_files(chunk_audio_files, final_audio_path)
            
            if success:
                # Clean up chunk files and silence files
                for audio_file in chunk_audio_files:
                    if audio_file.name.startswith(base_filename) or audio_file.name.startswith("silence_"):
                        try:
                            audio_file.unlink()
                        except:
                            pass  # Ignore cleanup errors
                
                logger.info(f"Successfully created concatenated audio: {final_audio_path}")
                return True, final_audio_path, chunk_info
            else:
                logger.error("Failed to concatenate audio files")
                return False, None, chunk_info
        
        logger.error("No audio files were generated")
        return False, None, chunk_info
        
    except Exception as e:
        logger.error(f"Error in chunked inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, []


if __name__ == "__main__":
    # Test the chunked inference (requires model setup)
    logging.basicConfig(level=logging.INFO)
    print("Chunked inference module ready for integration")