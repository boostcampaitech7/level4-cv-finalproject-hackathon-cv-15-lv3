import random
import nltk
from nltk.corpus import wordnet
from transformers import pipeline

# NLTK 데이터 다운로드 (최초 실행 시 필요)
nltk.download('wordnet')
nltk.download('omw-1.4')

# GPT 기반 Paraphrase Generator (Positive Augmentation)
paraphraser = pipeline("text2text-generation", model="t5-small")

def synonym_replacement(text, num_changes=1):
    """Positive Augmentation: Synonym Replacement (유의어 치환)"""
    words = text.split()
    new_words = words.copy()
    
    for _ in range(num_changes):
        word_idx = random.randint(0, len(words) - 1)
        synonyms = wordnet.synsets(words[word_idx])
        
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words[word_idx] = synonym.replace("_", " ")
    
    return " ".join(new_words)

def paraphrase_text(text):
    """Positive Augmentation: Paraphrasing (문장 구조 변형)"""
    outputs = paraphraser(text, max_length=50, num_return_sequences=1)
    return outputs[0]["generated_text"]

def add_random_noise(text, num_noises=1):
    """Negative Augmentation: Random Noise 추가"""
    words = text.split()
    noise_words = ["random", "unknown", "something", "somewhere", "thing"]
    
    for _ in range(num_noises):
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(noise_words))
    
    return " ".join(words)

def shuffle_words(text):
    """Negative Augmentation: Word Order Shuffle (단어 순서 랜덤 변경)"""
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def augment_query(query):
    """Query에 대해 Positive / Negative Augmentation 수행"""
    positive_aug1 = synonym_replacement(query)  # 유의어 치환
    positive_aug2 = paraphrase_text(query)  # 문장 구조 변형
    negative_aug1 = add_random_noise(query)  # 의미 무관 단어 추가
    negative_aug2 = shuffle_words(query)  # 단어 순서 랜덤 변경

    return {
        "original": query,
        "positive_aug": [positive_aug1, positive_aug2],
        "negative_aug": [negative_aug1, negative_aug2]
    }

# ✅ 테스트 예제
test_query = "A man is sitting on a bench in the park."
augmented_queries = augment_query(test_query)

print("Original:", augmented_queries["original"])
print("Positive Augmentations:", augmented_queries["positive_aug"])
print("Negative Augmentations:", augmented_queries["negative_aug"])