import requests

class DeepLTranslator:
    """DeepL API를 사용한 한국어 ↔ 영어 번역기 클래스"""
    
    def __init__(self):
        self.api_key = "dabf2942-070c-47e2-94e1-b43cbef766e3:fx"
        self.url = "https://api-free.deepl.com/v2/translate"

    def translate(self, text, source_lang, target_lang):
        """DeepL API를 사용하여 번역 수행"""
        params = {
            "auth_key": self.api_key,
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        response = requests.post(self.url, data=params)
        
        if response.status_code != 200:
            print(f"🚨 번역 API 오류: {response.status_code} - {response.text}")
            return None
        
        return response.json().get("translations", [{}])[0].get("text", "")

    def translate_ko_to_en(self, text):
        return self.translate(text, "KO", "EN")

    def translate_en_to_ko(self, text):
        return self.translate(text, "EN", "KO")