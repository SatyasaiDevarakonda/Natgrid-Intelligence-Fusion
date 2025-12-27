"""
NATGRID Configuration Management
Supports multiple LLM providers: Mistral API, Gemini, OpenAI, or Local GPU
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration"""
    
    # LLM Provider Selection
    llm_provider: str = "mistral_api"  # mistral_api, gemini, openai, local_gpu
    
    # API Keys
    mistral_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    # GPU Settings (for local_gpu only)
    use_gpu: bool = True
    use_4bit_quantization: bool = True
    mistral_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Other Models
    ner_model: str = "dslim/bert-base-NER"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Training Settings
    anomaly_contamination: float = 0.15
    anomaly_n_estimators: int = 100
    random_seed: int = 42
    
    # Paths
    project_dir: Path = None
    data_dir: Path = None
    models_dir: Path = None
    
    def __post_init__(self):
        if self.project_dir is None:
            self.project_dir = Path(__file__).parent
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
    
    def get_active_api_key(self) -> tuple:
        """Get the active API key based on provider"""
        if self.llm_provider == "mistral_api":
            return ("mistral_api", self.mistral_api_key)
        elif self.llm_provider == "gemini":
            return ("gemini", self.google_api_key)
        elif self.llm_provider == "openai":
            return ("openai", self.openai_api_key)
        elif self.llm_provider == "local_gpu":
            return ("local_gpu", self.huggingface_token)
        return (None, None)


def load_config() -> Config:
    """Load configuration from environment variables"""
    project_dir = Path(__file__).parent
    env_file = project_dir / ".env"
    
    if DOTENV_AVAILABLE and env_file.exists():
        load_dotenv(env_file)
        logger.info(f"✅ Loaded environment from {env_file}")
    elif not env_file.exists():
        logger.warning("⚠️  .env file not found! Copy .env.example to .env and add your API key")
    
    def parse_bool(value: str, default: bool = False) -> bool:
        if not value:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Get LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "mistral_api").lower()
    
    # Get API keys
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    
    # Clean up placeholder values
    if mistral_key and "your_" in mistral_key.lower():
        mistral_key = ""
    if google_key and "your_" in google_key.lower():
        google_key = ""
    if openai_key and "your_" in openai_key.lower():
        openai_key = ""
    if hf_token and "hf_xxx" in hf_token.lower():
        hf_token = ""
    
    # Validate that we have the required API key for the selected provider
    provider_key_map = {
        "mistral_api": ("MISTRAL_API_KEY", mistral_key, "https://console.mistral.ai/"),
        "gemini": ("GOOGLE_API_KEY", google_key, "https://makersuite.google.com/app/apikey"),
        "openai": ("OPENAI_API_KEY", openai_key, "https://platform.openai.com/api-keys"),
        "local_gpu": ("HUGGINGFACE_API_TOKEN", hf_token, "https://huggingface.co/settings/tokens")
    }
    
    if llm_provider in provider_key_map:
        key_name, key_value, url = provider_key_map[llm_provider]
        if not key_value:
            raise ValueError(
                f"\n{'='*60}\n"
                f"❌ {key_name} is required for {llm_provider} provider!\n\n"
                f"To fix this:\n"
                f"1. Get your API key from: {url}\n"
                f"2. Copy .env.example to .env\n"
                f"3. Add your key: {key_name}=your_key_here\n"
                f"{'='*60}"
            )
    
    config = Config(
        llm_provider=llm_provider,
        mistral_api_key=mistral_key or None,
        google_api_key=google_key or None,
        openai_api_key=openai_key or None,
        huggingface_token=hf_token or None,
        use_gpu=parse_bool(os.getenv("USE_GPU", "true"), True),
        use_4bit_quantization=parse_bool(os.getenv("USE_4BIT_QUANTIZATION", "true"), True),
        mistral_model=os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
        ner_model=os.getenv("NER_MODEL", "dslim/bert-base-NER"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        anomaly_contamination=float(os.getenv("ANOMALY_CONTAMINATION", "0.15")),
        anomaly_n_estimators=int(os.getenv("ANOMALY_N_ESTIMATORS", "100")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        project_dir=project_dir
    )
    
    logger.info(f"✅ Using LLM Provider: {config.llm_provider}")
    
    return config


def check_gpu_availability() -> dict:
    """Check GPU availability"""
    info = {
        'cuda_available': False,
        'device_count': 0,
        'devices': [],
        'recommended_device': 'cpu'
    }
    
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        
        if info['cuda_available']:
            info['device_count'] = torch.cuda.device_count()
            for i in range(info['device_count']):
                info['devices'].append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                })
            info['recommended_device'] = 'cuda:0'
    except ImportError:
        pass
    
    return info


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    print("=" * 60)
    print("NATGRID Configuration Test")
    print("=" * 60)
    
    try:
        config = load_config()
        print(f"\n✅ Configuration loaded!")
        print(f"   LLM Provider: {config.llm_provider}")
        print(f"   Project Dir: {config.project_dir}")
        
        provider, key = config.get_active_api_key()
        print(f"   API Key: {'***' + key[-4:] if key else 'Not set'}")
        
    except ValueError as e:
        print(str(e))
        sys.exit(1)
