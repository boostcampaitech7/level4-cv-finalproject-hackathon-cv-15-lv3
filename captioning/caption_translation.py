from googletrans import Translator
from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer
from google.cloud import translate_v2 as translate
import deepl
from translate import Translator
from azure.core.credentials import AzureKeyCredential
# from azure.ai.translation.text import TextTranslationClient

import json

# googletrans 사용

# # 번역기 객체 생성
# translator = Translator()

# # JSON 파일 읽기
# with open("updated_Movieclips_annotations.json", "r") as f:
#     data = json.load(f)

# # 영어 캡션 → 한국어 번역
# for item in data:
#     translation = translator.translate(item["caption"], src="en", dest="ko")
#     item["caption_kr"] = translation.text  # 번역 결과 저장

# # JSON 파일 저장
# with open("captions_translated_googletrans.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)


# transformer 번역 모델 사용

# 모델과 토크나이저 로드
model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

tokenizer.tgt_lang = "ko"

# JSON 파일 읽기
with open("updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 영어 캡션 → 한국어 번역
for item in data:
    # 영어 캡션을 한국어로 번역
    inputs = tokenizer([item["caption"]], return_tensors="pt")

    # 모델에 언어 코드 지정하여 번역
    translated = model.generate(**inputs)
    
    # 번역된 텍스트 저장
    item["caption_kr"] = tokenizer.decode(translated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# JSON 파일 저장
with open("captions_translated_transformers.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


# Google Cloud Translate 사용=> 다시 시도

# # Google Cloud Translate Client 생성
# client = translate.Client()

# # JSON 파일 읽기
# with open("updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 영어 캡션 → 한국어 번역
# for item in data:
#     result = client.translate(item["caption"], source_language="en", target_language="ko")
#     item["caption_kr"] = result['translatedText']  # 번역 결과 저장

# # JSON 파일 저장
# with open("captions_translated_gct.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)


# DeepL 사용

# # DeepL Translator Client 생성
# auth_key = "70c47127-a731-47c8-a901-69f325feacf7:fx"  # DeepL API 키
# translator = deepl.Translator(auth_key)

# # JSON 파일 읽기
# with open("updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 영어 캡션 → 한국어 번역
# for item in data:
#     translation = translator.translate_text(item["caption"], source_lang="EN", target_lang="KO")
#     item["caption_kr"] = translation.text  # 번역 결과 저장

# # JSON 파일 저장
# with open("captions_translated_deepl.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)


# translate 라이브러리 사용

# # 번역기 객체 생성 (영어 → 한국어)
# translator = Translator(from_lang="en", to_lang="ko")

# # JSON 파일 읽기
# with open("updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 영어 캡션 → 한국어 번역
# for item in data:
#     item["caption_kr"] = translator.translate(item["caption"])  # 번역 결과 저장

# # JSON 파일 저장
# with open("captions_translated_translateL.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)


# Microsoft Azure Translator 사용=> 다시 시도

# # Azure Translation API 설정
# endpoint = "https://api.cognitive.microsofttranslator.com/"  # Azure Translation API 엔드포인트
# api_key = "2WkiazCmaUbWxAPuqkWavGYFmZ3yhGHUEPtJeQ2MVCTfTW2z9OC5JQQJ99BAACNns7RXJ3w3AAAbACOGO8Gp"    # 발급받은 API 키

# # 번역 클라이언트 생성
# credential = AzureKeyCredential(api_key)
# translator_client = TextTranslationClient(endpoint=endpoint, credential=credential)

# # JSON 파일 읽기
# with open("updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 영어 캡션 → 한국어 번역
# for item in data:
#     response = translator_client.translate(content=item["caption"], to=["ko"])
#     item["caption_kr"] = response[0].translations[0].text  # 번역 결과 저장

# # JSON 파일 저장
# with open("captions_translated_azure.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)