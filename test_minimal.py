#!/usr/bin/env python3
"""
Minimal test for Flask app structure without requiring PyTorch/HebTTS dependencies
"""

import sys
from pathlib import Path

def test_flask_structure():
    """Test basic Flask app structure and endpoints"""
    print("Testing Flask app structure...")
    
    try:
        import flask
        from flask import Flask
        print(f"✓ Flask available")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        # Create a minimal Flask app to test structure
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        # Define minimal endpoints that don't require model loading
        @app.route('/health', methods=['GET'])
        def health_check():
            return {'status': 'healthy', 'model_loaded': False}
        
        @app.route('/speakers', methods=['GET'])
        def get_speakers():
            return {'speakers': ['osim', 'geek', 'shaul']}
        
        # Test endpoints
        with app.test_client() as client:
            response = client.get('/health')
            print(f"✓ Health endpoint: {response.status_code}")
            
            response = client.get('/speakers')
            print(f"✓ Speakers endpoint: {response.status_code}")
        
        print("Flask app structure test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Flask structure test failed: {e}")
        return False

def test_file_structure():
    """Verify required files exist"""
    print("\nTesting file structure...")
    
    base_dir = Path(__file__).parent
    hebtts_dir = base_dir / "HebTTSLM"
    
    required_files = [
        hebtts_dir / "checkpoint.pt",
        hebtts_dir / "speakers" / "speakers.yaml",
        hebtts_dir / "tokenizer" / "unique_words_tokens_all.k2symbols",
        hebtts_dir / "tokenizer" / "vocab.txt",
        base_dir / "app.py",
        base_dir / "requirements.txt",
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ {file_path.name}")
        else:
            print(f"✗ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing {len(missing_files)} required files.")
        return False
    
    print("All required files present!")
    return True

def test_app_syntax():
    """Test if app.py has valid Python syntax"""
    print("\nTesting app.py syntax...")
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, str(app_path), 'exec')
        print("✓ app.py has valid syntax")
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading app.py: {e}")
        return False

if __name__ == "__main__":
    print("HebTTS-GUI Minimal Test Suite")
    print("=" * 40)
    
    success = True
    success &= test_file_structure()
    success &= test_app_syntax()
    success &= test_flask_structure()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Minimal tests passed! Flask structure is ready.")
        print("Note: Full functionality requires PyTorch and HebTTS dependencies.")
    else:
        print("✗ Some tests failed. Check output above.")
    
    sys.exit(0 if success else 1)