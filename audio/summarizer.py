"""
TrueVision — Local Extractive Summarizer

Simple fallback summarizer used when the server/LLM is unavailable.
"""

def summarize_extractive(text: str, max_sentences: int = 3) -> str:
    """
    Extract the first N sentences from a text block.
    """
    if not text:
        return ""
        
    sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
    if not sentences:
        return text
        
    extracted = sentences[:max_sentences]
    return ". ".join(extracted) + "."


def summarize_one_sentence(text: str, max_chars: int = 140) -> str:
    """
    Extracts the first sentence and clamps it to max_chars.
    Clips at word boundaries with an ellipsis if needed.
    """
    if not text:
        return ""
        
    first_sentence = text.replace('?', '.').replace('!', '.').split('.')[0].strip()
    
    if len(first_sentence) <= max_chars:
        return first_sentence + "."
        
    # Truncate to max_chars and find last space
    truncated = first_sentence[:max_chars - 3]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
        
    return truncated + "..."
