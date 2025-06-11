"""reward_language.py – English-only bonus/penalty for GRPO-style RLHF."""
import re
import os

# NumPy 2.0 compatibility fix for FastText
try:
    import numpy as np
    if hasattr(np, '__version__') and np.__version__.startswith('2.'):
        # Monkey patch numpy to handle FastText's copy=False issue
        original_array = np.array
        def patched_array(*args, **kwargs):
            if 'copy' in kwargs and kwargs['copy'] is False:
                kwargs['copy'] = None  # Let NumPy decide
            return original_array(*args, **kwargs)
        np.array = patched_array
except ImportError:
    pass

import fasttext

# ----------  Normalisation --------------------------------------------------
_LATEX   = re.compile(r'\$.*?\$|\\\[.*?\\\]|\\begin\{.*?\}.*?\\end\{.*?\}', re.S)
_CODE    = re.compile(r'```.*?```', re.S)

def strip_noise(text: str) -> str:
    """Remove LaTeX and fenced-code so the language classifier stays honest."""
    return re.sub(r'\s+', ' ',
                  _CODE.sub(' ', _LATEX.sub(' ', text))).strip()

# ----------  Language detection ---------------------------------------------
# Global model instance - loaded once
_FASTTEXT_MODEL = None

def _get_model():
    """Get the fasttext model, loading it once on first call."""
    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is None:
        # Look for the model in the current working directory or relative to this file
        model_paths = [
            'lid.176.bin',  # In working directory
            os.path.join(os.path.dirname(__file__), '..', 'lid.176.bin'),  # In parent directory
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "Could not find lid.176.bin model file. Please download it with:\n"
                "curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
        
        _FASTTEXT_MODEL = fasttext.load_model(model_path)
    
    return _FASTTEXT_MODEL

def detect_lang(text: str, k: int = 1) -> str:
    """Return ISO-639-1 language code predicted with highest probability."""
    # FastText predict requires single line text - replace newlines with spaces
    single_line_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    if not single_line_text:  # Handle empty text - default to English for backward compatibility
        return 'en'
    
    label, _ = _get_model().predict(single_line_text, k=k)
    return label[0].replace('__label__', '')

# ----------  Reward ----------------------------------------------------------
def language_reward(problem: str,
                    thoughts: str,
                    answer: str,
                    target: str = 'en',
                    bonus: float = 0.1,
                    penalty: float = -0.2) -> float:
    """
    +bonus  if all three parts are English;
    negative penalty proportional to #non-English parts otherwise.
    """
    parts   = [problem, thoughts, answer]
    langs   = [detect_lang(strip_noise(t or '')) for t in parts]
    off_cnt = sum(l != target for l in langs)
    return bonus if off_cnt == 0 else penalty * off_cnt / len(parts)

# ----------  Optional fast path ---------------------------------------------
_NON_LATIN = re.compile(r'[\u0400-\u052F\u2DE0-\u2DFF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')  # Cyrillic + Hiragana + Katakana + CJK

def quick_check(text: str) -> bool:
    """True iff text contains only (extended) Latin; false triggers penalty."""
    return _NON_LATIN.search(text) is None
