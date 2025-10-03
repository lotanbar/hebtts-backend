#!/usr/bin/env python3
"""
Simple test script to verify Flask app imports and basic functionality
without requiring full model loading (for development testing)
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import flask_cors
        print("✓ Flask-CORS")
    except ImportError as e:
        print(f"✗ Flask-CORS import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✓ TorchAudio {torchaudio.__version__}")
    except ImportError as e:
        print(f"✗ TorchAudio import failed: {e}")
        return False
    
    try:
        from omegaconf import OmegaConf
        print("✓ OmegaConf")
    except ImportError as e:
        print(f"✗ OmegaConf import failed: {e}")
        return False
    
    print("All core imports successful!")
    return True

def test_hebtts_imports():
    """Test HebTTSLM imports"""
    print("\nTesting HebTTSLM imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "HebTTSLM"))
        from HebTTSLM.infer import prepare_inference, infer_texts
        from HebTTSLM.utils import AttributeDict
        print("✓ HebTTSLM imports successful")
        return True
    except ImportError as e:
        print(f"✗ HebTTSLM import failed: {e}")
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
        print("Note: checkpoint.pt is large (~1-2GB) and may need to be downloaded separately.")
        return False
    
    print("All required files present!")
    return True

def test_flask_app_creation():
    """Test Flask app can be created (without model loading)"""
    print("\nTesting Flask app creation...")
    
    try:
        # Import our app module but don't load the model
        import handler
        
        # Create a test client
        handler.app.config['TESTING'] = True
        with handler.app.test_client() as client:
            # Test health endpoint (should work even without model)
            response = client.get('/health')
            print(f"✓ Health endpoint responds: {response.status_code}")
            
            # Test speakers endpoint
            response = client.get('/speakers')
            print(f"✓ Speakers endpoint responds: {response.status_code}")
        
        print("Flask app creation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False

if __name__ == "__main__":
    print("HebTTS-GUI Test Suite")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_hebtts_imports()
    success &= test_file_structure()
    success &= test_flask_app_creation()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Ready for deployment.")
    else:
        print("✗ Some tests failed. Check output above.")
    
    sys.exit(0 if success else 1)