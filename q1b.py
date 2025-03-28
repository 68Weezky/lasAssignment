
from googletrans import Translator

# Initialize the translator
translator = Translator()

def translate_text(text, src_lang='en', dest_lang='sw'):
    """
    Translate text from source language to destination language.

    Parameters:
        text (str): The text to translate.
        src_lang (str): Source language code (default: 'en' for English).
        dest_lang (str): Destination language code (default: 'sw' for Swahili).

    Returns:
        str: Translated text.
    """
    try:
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
        return translation.text
    except Exception as e:
        return f"Translation failed: {e}"

# Test the translation
english_text = "Good morning?"
swahili_translation = translate_text(english_text, src_lang='en', dest_lang='sw')
print(f"English: {english_text}\nSwahili: {swahili_translation}\n")

swahili_text = "Mwalimu ameingia darasani?"
english_translation = translate_text(swahili_text, src_lang='sw', dest_lang='en')
print(f"Swahili: {swahili_text}\nEnglish: {english_translation}\n")
