class DeepLTranslator:
    """DeepL API를 사용한 한국어 ↔ 영어 번역기 클래스"""
    
    def __init__(self, api_key):
        self.api_key = api_key
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


from translate import Translator as TranslateLib

class Translator:
    """
    translate 라이브러리를 사용한 한국어 ↔ 영어 번역기 클래스
    """

    def __init__(self, source_lang: str = "ko", target_lang: str = "en"):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def _split_text(self, text: str, max_length: int = 500) -> list:
        """문장을 최대 길이(max_length) 이하의 조각으로 나누는 함수"""
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    def _translate_chunk(self, text: str, from_lang: str, to_lang: str) -> str:
        """500자 이하의 텍스트를 번역하는 함수"""
        try:
            translator = TranslateLib(from_lang=from_lang, to_lang=to_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"🚨 번역 요청 중 오류 발생: {e}")
            return ""

    def translate(self, text: str) -> str:
        """긴 텍스트를 자동으로 나누어 번역 후 합치는 함수"""
        chunks = self._split_text(text)
        translated_chunks = [self._translate_chunk(chunk, self.source_lang, self.target_lang) for chunk in chunks]
        return " ".join(translated_chunks)

    def translate_ko_to_en(self, text: str) -> str:
        """한국어 -> 영어 번역"""
        self.source_lang, self.target_lang = "ko", "en"
        return self.translate(text)

    def translate_en_to_ko(self, text: str) -> str:
        """영어 -> 한국어 번역"""
        self.source_lang, self.target_lang = "en", "ko"
        return self.translate(text)

import requests
from concurrent.futures import ThreadPoolExecutor

# 멀티스레딩을 지원하는 병렬 번역기
class ParallelTranslator:
    """
    멀티스레딩을 활용한 병렬 번역 클래스
    """

    def __init__(self, translator, max_workers=4):
        """
        translator: DeepLTranslator 또는 Translator 인스턴스
        max_workers: 병렬 처리할 번역 요청의 개수 (기본값 4)
        """
        self.translator = translator
        self.max_workers = max_workers

    def batch_translate(self, texts, direction="ko_to_en"):
        """멀티스레딩을 활용한 병렬 번역"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if direction == "ko_to_en":
                results = list(executor.map(self.translator.translate_ko_to_en, texts))
            else:
                results = list(executor.map(self.translator.translate_en_to_ko, texts))
        return results


    # ✅ translator_mode를 설정하여 선택적으로 번역기 사용 가능
    def get_translator(translator_mode, api_key=None, max_workers=4):
        """
        translator_mode:
        - "deepl"  : DeepL 번역기 사용
        - "translate"  : translate 라이브러리 사용
        - "parallel-deepl"  : DeepL 병렬 번역
        - "parallel-translate"  : translate 병렬 번역
        """
        if translator_mode == "deepl":
            return DeepLTranslator(api_key=api_key)
        elif translator_mode == "translate":
            return Translator()
        elif translator_mode == "parallel-deepl":
            return ParallelTranslator(DeepLTranslator(api_key=api_key), max_workers=max_workers)
        elif translator_mode == "parallel-translate":
            return ParallelTranslator(Translator(), max_workers=max_workers)
        else:
            raise ValueError(f"🚨 지원되지 않는 translator_mode: {translator_mode}")

from deep_translator import GoogleTranslator

class DeepGoogleTranslator:
    """deep-translator 라이브러리를 사용한 한국어 ↔️ 영어 번역기 클래스"""
    
    def __init__(self):
        self.ko_to_en = GoogleTranslator(source='ko', target='en')
        self.en_to_ko = GoogleTranslator(source='en', target='ko')
    def translate_ko_to_en(self, text):
        """한국어 → 영어 번역"""
        try:
            return self.ko_to_en.translate(text)
        except Exception as e:
            print(f"🚨 번역 오류: {str(e)}")
            return None
    def translate_en_to_ko(self, text):
        """영어 → 한국어 번역"""
        try:
            return self.en_to_ko.translate(text)
        except Exception as e:
            print(f"🚨 번역 오류: {str(e)}")
            return None