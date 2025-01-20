import deepl
import json

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def translate_captions(data, auth_key):
    """Translate captions from English to Korean using DeepL."""
    translator = deepl.Translator(auth_key)
    for item in data:
        translation = translator.translate_text(item["caption"], source_lang="EN", target_lang="KO")
        item["caption_kr"] = translation.text
    return data

def clean_caption(caption):
    """Clean and format a caption."""
    sentences = [s.strip() for s in caption.split(".") if s.strip()]
    if sentences and not sentences[-1].endswith((".", "?", "!")):
        sentences.pop()
    return ". ".join(sentences) + "." if sentences else caption

def clean_captions(data):
    """Clean all captions in the data."""
    for item in data:
        item["caption"] = clean_caption(item["caption"])
    return data

def main():
    # File paths
    input_file = "../json/updated_Movieclips_annotations.json"
    cleaned_file = "../json/clean_Movieclips_annotations.json"
    translated_file = "../json/translated_Movieclips_annotations.json"

    # Load and clean captions
    data = load_json(input_file)
    data = clean_captions(data)
    save_json(data, cleaned_file)

    # Translate captions
    auth_key = "70c47127-a731-47c8-a901-69f325feacf7:fx"  # DeepL API key
    translated_data = translate_captions(data, auth_key)
    save_json(translated_data, translated_file)

if __name__ == "__main__":
    main()
