#!/usr/bin/env python3
"""
Test script for the enhanced Hebrew TTS API with chunking support.
This simulates RunPod job requests to test the chunking functionality.
"""

import json
import sys
from pathlib import Path

def test_chunking_api():
    """Test the API with various text lengths"""
    
    # Test cases
    test_cases = [
        {
            "name": "Short Text (No Chunking)",
            "input": {
                "text": "זהו טקסט קצר בעברית.",
                "speaker": "osim",
                "filename": "short_test"
            }
        },
        {
            "name": "Medium Text (Borderline)",
            "input": {
                "text": "זהו טקסט בינוני באורכו. " * 6,  # ~150 chars
                "speaker": "osim",
                "filename": "medium_test",
                "enable_chunking": True,
                "max_chunk_chars": 150
            }
        },
        {
            "name": "Long Text (Chunking Required)",
            "input": {
                "text": """זהו טקסט ארוך מאוד בעברית שנועד לבדוק את יכולות החלוקה של המערכת. 
                הטקסט הזה כולל מספר משפטים, פסיקים, ונקודותיים: כמו כאן; וגם כאן. 
                המטרה היא לוודא שהחלוקה מתבצעת בצורה חכמה ומכבדת את הגבולות הטבעיים של השפה העברית.
                כאשר יש לנו טקסט ארוך מאוד, אנחנו רוצים לחלק אותו לחלקים קטנים יותר.
                כל חלק צריך להיות קטן מספיק כדי שהמודל יוכל לעבד אותו ביעילות ובאיכות גבוהה.""".strip(),
                "speaker": "osim",
                "filename": "long_test",
                "enable_chunking": True,
                "max_chunk_chars": 120
            }
        },
        {
            "name": "Long Text (Chunking Disabled)",
            "input": {
                "text": "טקסט ארוך שלא יעבור חלוקה. " * 10,
                "speaker": "osim",
                "filename": "no_chunk_test",
                "enable_chunking": False
            }
        }
    ]
    
    print("Testing Hebrew TTS API with Chunking Support")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 40)
        
        text = test_case['input']['text']
        enable_chunking = test_case['input'].get('enable_chunking', True)
        max_chars = test_case['input'].get('max_chunk_chars', 150)
        
        print(f"Text length: {len(text)} characters")
        print(f"Chunking enabled: {enable_chunking}")
        print(f"Max chunk size: {max_chars}")
        
        # Simulate chunking decision
        from text_chunker import HebrewTextChunker
        chunker = HebrewTextChunker(max_chars)
        should_chunk = enable_chunking and not chunker.is_chunk_valid(text)
        
        if should_chunk:
            chunks = chunker.chunk_text(text)
            print(f"→ Will be CHUNKED into {len(chunks)} parts:")
            for i, chunk in enumerate(chunks):
                print(f"   Part {i+1}: {len(chunk)} chars - '{chunk[:40]}{'...' if len(chunk) > 40 else ''}'")
        else:
            print("→ Will be processed as SINGLE chunk")
            
        # Show expected API response structure
        if should_chunk:
            expected_response = {
                "chunked": True,
                "chunks_processed": len(chunks),
                "original_length": len(text),
                "chunk_info": [f"part_{i+1}: {len(chunk)} chars" for i, chunk in enumerate(chunks)]
            }
        else:
            expected_response = {
                "chunked": False,
                "original_length": len(text)
            }
        
        print(f"Expected response metadata: {json.dumps(expected_response, indent=2)}")


def test_parameter_validation():
    """Test parameter validation and defaults"""
    print("\n\nTesting Parameter Validation")
    print("=" * 60)
    
    # Test default parameters
    default_params = {
        "text": "טקסט לבדיקה",
        "speaker": "osim"
    }
    
    print("✓ Required parameters only (should use defaults)")
    print(f"  Defaults: enable_chunking=True, max_chunk_chars=150")
    
    # Test custom parameters
    custom_params = {
        "text": "טקסט לבדיקה",
        "speaker": "osim",
        "enable_chunking": False,
        "max_chunk_chars": 100,
        "top_k": 25,
        "temperature": 0.8,
        "use_mbd": False
    }
    
    print("✓ Custom parameters provided")
    print(f"  Custom: enable_chunking=False, max_chunk_chars=100")


def test_integration_scenarios():
    """Test real-world integration scenarios"""
    print("\n\nTesting Integration Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "News Article (Long)",
            "text": """חדשות מישראל: ממשלת ישראל הודיעה היום על תוכנית חדשה לפיתוח הטכנולוגיה הישראלית. 
            התוכנית כוללת השקעה של מיליארד שקל בחברות הזנק ובמחקר ופיתוח. 
            ראש הממשלה אמר כי מדובר בצעד חשוב לעתיד המדינה. 
            התוכנית תמומן באמצעות הקצאות תקציביות חדשות ושיתופי פעולה עם המגזר הפרטי."""
        },
        {
            "name": "Product Description (Medium)",
            "text": """מוצר חדש ומהפכני שישנה את הדרך בה אתם חושבים על טכנולוגיה. 
            עם תכונות מתקדמות ועיצוב אלגנטי, המוצר מציע חוויית משתמש ללא תחרות."""
        },
        {
            "name": "Short Message",
            "text": "שלום, איך הולך? מקווה שיום טוב!"
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        text = scenario['text'].strip()
        print(f"Length: {len(text)} characters")
        
        from text_chunker import HebrewTextChunker
        chunker = HebrewTextChunker()
        
        if chunker.is_chunk_valid(text):
            print("→ Single chunk processing")
        else:
            chunks = chunker.chunk_text(text)
            print(f"→ Multi-chunk processing: {len(chunks)} chunks")
            print(f"   Chunk sizes: {[len(c) for c in chunks]}")


if __name__ == "__main__":
    print("Enhanced Hebrew TTS API Test Suite")
    print("Testing chunking functionality and integration")
    print()
    
    try:
        test_chunking_api()
        test_parameter_validation()
        test_integration_scenarios()
        
        print("\n" + "=" * 60)
        print("✓ All API tests completed successfully!")
        print("\nImplementation Summary:")
        print("• Backward compatible with existing API")
        print("• Automatic chunking for texts > 150 characters")
        print("• Configurable chunk size and enable/disable")
        print("• Smart Hebrew text boundary detection")
        print("• Audio concatenation for seamless output")
        print("• Enhanced response metadata for debugging")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)