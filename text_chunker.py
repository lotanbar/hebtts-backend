"""
Hebrew Text Chunking Utilities for TTS Processing

This module provides intelligent text chunking for Hebrew TTS models,
respecting model limitations while maintaining natural speech boundaries.
"""

import re
import logging
from typing import List, Tuple
from pathlib import Path


class HebrewTextChunker:
    """
    Handles chunking of Hebrew text for TTS processing.
    
    Respects model limitations:
    - Tokenizer max length: 512 tokens
    - Training data optimal: ~150-200 characters
    - Uses smart boundary detection for natural speech
    """
    
    def __init__(self, max_chars: int = 150):
        """
        Initialize the chunker.
        
        Args:
            max_chars: Maximum characters per chunk (default: 150)
        """
        self.max_chars = max_chars
        self.logger = logging.getLogger(__name__)
        
        # Hebrew sentence boundaries - common punctuation
        self.sentence_ends = re.compile(r'[.!?״׳][\s]*')
        
        # Hebrew clause boundaries - pauses in speech
        self.clause_boundaries = re.compile(r'[,;:–—־][\s]*')
        
        # Word boundaries for fallback
        self.word_boundary = re.compile(r'\s+')
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for Hebrew text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens (conservative estimate)
        """
        # Conservative estimation: Hebrew averages ~2 characters per token
        # but we use 1.5 to be safe with the 512 token limit
        return int(len(text) / 1.5)
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentence boundaries.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences with preserved punctuation
        """
        # Split while preserving the delimiters
        parts = self.sentence_ends.split(text)
        sentences = []
        
        for i, part in enumerate(parts):
            if part.strip():
                sentences.append(part.strip())
        
        return sentences
    
    def split_by_clauses(self, text: str) -> List[str]:
        """
        Split text by clause boundaries (commas, semicolons, etc.).
        
        Args:
            text: Input text
            
        Returns:
            List of clauses with preserved punctuation
        """
        parts = self.clause_boundaries.split(text)
        clauses = []
        
        for part in parts:
            if part.strip():
                clauses.append(part.strip())
                
        return clauses
    
    def split_by_words(self, text: str, max_words: int = 20) -> List[str]:
        """
        Split text by word boundaries as final fallback.
        
        Args:
            text: Input text
            max_words: Maximum words per chunk
            
        Returns:
            List of word chunks
        """
        words = self.word_boundary.split(text.strip())
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def is_chunk_valid(self, text: str) -> bool:
        """
        Check if a text chunk is valid for processing.
        
        Args:
            text: Text chunk to validate
            
        Returns:
            True if chunk is valid, False otherwise
        """
        return (len(text.strip()) > 0 and 
                len(text) <= self.max_chars and 
                self.estimate_token_count(text) <= 400)  # Safety margin under 512
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Intelligently chunk text respecting natural boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks suitable for TTS processing
        """
        text = text.strip()
        if not text:
            return []
        
        self.logger.info(f"Chunking text of {len(text)} characters")
        
        # If text is already valid, return as is
        if self.is_chunk_valid(text):
            return [text]
        
        chunks = []
        
        # Strategy 1: Split by sentences
        sentences = self.split_by_sentences(text)
        current_chunk = ""
        
        for sentence in sentences:
            # If single sentence is too long, split it further
            if len(sentence) > self.max_chars:
                # Save current chunk if exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Strategy 2: Split long sentence by clauses
                clauses = self.split_by_clauses(sentence)
                
                for clause in clauses:
                    # If clause is still too long, split by words
                    if len(clause) > self.max_chars:
                        word_chunks = self.split_by_words(clause)
                        
                        for word_chunk in word_chunks:
                            if current_chunk and len(current_chunk + " " + word_chunk) > self.max_chars:
                                chunks.append(current_chunk.strip())
                                current_chunk = word_chunk
                            else:
                                current_chunk = (current_chunk + " " + word_chunk).strip()
                    else:
                        # Check if we can add clause to current chunk
                        potential_chunk = (current_chunk + " " + clause).strip()
                        if len(potential_chunk) <= self.max_chars:
                            current_chunk = potential_chunk
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = clause
            else:
                # Try to add sentence to current chunk
                potential_chunk = (current_chunk + " " + sentence).strip()
                if len(potential_chunk) <= self.max_chars:
                    current_chunk = potential_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter and validate chunks
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        
        self.logger.info(f"Split into {len(valid_chunks)} chunks")
        for i, chunk in enumerate(valid_chunks):
            self.logger.debug(f"Chunk {i+1}: {len(chunk)} chars, ~{self.estimate_token_count(chunk)} tokens")
        
        return valid_chunks


def prepare_chunked_texts(text: str, base_filename: str = "output", max_chars: int = 150) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Prepare text chunks with filenames for TTS processing.
    
    Args:
        text: Input text to process
        base_filename: Base filename for outputs
        max_chars: Maximum characters per chunk
        
    Returns:
        Tuple of (list of (filename, text) pairs, is_chunked_flag)
    """
    chunker = HebrewTextChunker(max_chars)
    chunks = chunker.chunk_text(text)
    
    if len(chunks) <= 1:
        # Single chunk or no chunking needed
        return [(base_filename, chunks[0] if chunks else text)], False
    else:
        # Multiple chunks - add sequence numbers
        texts_with_filenames = []
        for i, chunk in enumerate(chunks):
            filename = f"{base_filename}_part_{i+1:03d}_of_{len(chunks):03d}"
            texts_with_filenames.append((filename, chunk))
        
        return texts_with_filenames, True


if __name__ == "__main__":
    # Test the chunker
    logging.basicConfig(level=logging.INFO)
    
    sample_text = """
    זהו טקסט ארוך מאוד בעברית שנועד לבדוק את יכולות החלוקה של המערכת. 
    הטקסט הזה כולל מספר משפטים, פסיקים, ונקודותיים: כמו כאן; וגם כאן. 
    המטרה היא לוודא שהחלוקה מתבצעת בצורה חכמה ומכבדת את הגבולות הטבעיים של השפה העברית.
    כאשר יש לנו טקסט ארוך מאוד, אנחנו רוצים לחלק אותו לחלקים קטנים יותר.
    """
    
    texts_with_filenames, is_chunked = prepare_chunked_texts(sample_text.strip())
    
    print(f"Original text: {len(sample_text)} characters")
    print(f"Chunked: {is_chunked}")
    print(f"Number of parts: {len(texts_with_filenames)}")
    
    for filename, chunk in texts_with_filenames:
        print(f"\n{filename} ({len(chunk)} chars):")
        print(f"  {chunk[:100]}{'...' if len(chunk) > 100 else ''}")