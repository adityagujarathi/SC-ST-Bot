"""
Language utilities for the SC/ST Bot application.
Provides language detection and translation capabilities.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Union
import logging

# Try different translation libraries in order of preference
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR = "deep_translator"
except ImportError:
    try:
        from googletrans import Translator
        translator = Translator()
        TRANSLATOR = "googletrans"
    except ImportError:
        TRANSLATOR = None

try:
    from langdetect import detect, LangDetectException
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported languages with their codes and names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "or": "Odia"
}

# Path to store translations cache
TRANSLATIONS_CACHE_FILE = "translations_cache.json"

# Initialize translations cache
translations_cache = {}
if os.path.exists(TRANSLATIONS_CACHE_FILE):
    try:
        with open(TRANSLATIONS_CACHE_FILE, 'r', encoding='utf-8') as f:
            translations_cache = json.load(f)
    except Exception as e:
        logger.error(f"Error loading translations cache: {str(e)}")

def save_translations_cache():
    """Save translations to cache file"""
    try:
        with open(TRANSLATIONS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(translations_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving translations cache: {str(e)}")

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    
    Args:
        text: Text to detect language for
        
    Returns:
        Language code (e.g., 'en', 'hi')
    """
    if not LANG_DETECT_AVAILABLE:
        return "en"  # Default to English if langdetect is not available
    
    if not text or len(text.strip()) < 10:
        return "en"  # Default to English for very short texts
    
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except LangDetectException:
        return "en"  # Default to English on detection failure

def translate_text(text: str, target_lang: str = "en", source_lang: Optional[str] = None) -> str:
    """
    Translate text to the target language.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code (if None, will be auto-detected)
        
    Returns:
        Translated text
    """
    if not text or not text.strip():
        return text
    
    if not TRANSLATOR:
        return text  # Return original text if no translator is available
    
    # If target language is English and source is not specified, detect language
    if target_lang == "en" and not source_lang and LANG_DETECT_AVAILABLE:
        source_lang = detect_language(text)
        
        # If detected language is already English, return the original text
        if source_lang == "en":
            return text
    
    # If source and target are the same, return original text
    if source_lang and source_lang == target_lang:
        return text
    
    # Create cache key
    cache_key = f"{source_lang or 'auto'}|{target_lang}|{text}"
    
    # Check if translation is in cache
    if cache_key in translations_cache:
        return translations_cache[cache_key]
    
    try:
        # Translate using the available translator
        if TRANSLATOR == "deep_translator":
            translator = GoogleTranslator(source=source_lang or 'auto', target=target_lang)
            result = translator.translate(text)
        elif TRANSLATOR == "googletrans":
            result = translator.translate(text, dest=target_lang, src=source_lang or 'auto').text
        else:
            return text
        
        # Cache the result
        translations_cache[cache_key] = result
        save_translations_cache()
        
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text on error

def get_ui_text(lang_code: str = "en") -> Dict[str, str]:
    """
    Get UI text in the specified language.
    
    Args:
        lang_code: Language code
        
    Returns:
        Dictionary of UI text elements
    """
    # Define UI text in English
    ui_text_en = {
        "app_title": "SC/ST Atrocities Act Research Assistant",
        "app_description": "This research assistant uses AI to help frontline justice workers with cases related to the Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989.",
        "settings": "Settings",
        "language": "Language",
        "api_key_missing": "API key not found. Please add your Gemini API key to the .env file.",
        "api_key_success": "✅ Gemini API configured successfully",
        "enhanced_search_available": "✅ Enhanced search is available",
        "enhanced_search_unavailable": "❌ Enhanced search is not available",
        "install_scikit": "Install scikit-learn for better search results",
        "document_processing": "Document Processing",
        "reprocess_button": "Reprocess All Documents",
        "reprocessing_message": "Reprocessing all documents. This may take a while...",
        "reprocessing_success": "Documents reprocessed successfully!",
        "table_extraction_available": "✅ Table extraction is available",
        "table_extraction_unavailable": "Tabula-py is not installed. Table extraction from PDFs will be limited.",
        "document_statistics": "Document Statistics",
        "total_chunks": "Total document chunks",
        "table_chunks": "Table chunks",
        "unique_sources": "Unique source files",
        "pdf_files": "PDF files",
        "docx_files": "DOCX files",
        "pptx_files": "PPTX files",
        "processing_log": "Processing Log",
        "question_prompt": "Ask a question about the SC/ST Atrocities Act, Rules, or related legal matters:",
        "your_question": "Your question:",
        "submit_button": "Submit",
        "researching": "Researching your question...",
        "about_app": "About this Application",
        "translate_response": "Translate response to:",
        "original_language": "Original"
    }
    
    # If language is English, return English text
    if lang_code == "en":
        return ui_text_en
    
    # For other languages, translate each UI text element
    ui_text_translated = {}
    for key, text in ui_text_en.items():
        ui_text_translated[key] = translate_text(text, target_lang=lang_code, source_lang="en")
    
    return ui_text_translated
