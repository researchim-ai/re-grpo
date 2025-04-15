"""
This script performs two main tasks:
1. It loads a markdown document, splits it into chunks, generates embeddings,
   and builds a FAISS index (which is saved locally).
2. It generates QA pairs from the document using Qwen model.
   For each chunk (using a sliding window for context), it generates multiple question-answer pairs
   with different difficulties. The generation is performed in batch with one retry for failed prompts.
   Successfully generated QA pairs are saved to "qa_generated_data/questions.json".

Requirements:
    pip install langchain faiss-cpu transformers
"""

import os
import torch
import re
import json
import pickle
from typing import List, Tuple, Optional, Dict
import gc
import time

# ========= Part 1: Document Processing and Embedding Generation =========

# Load and split the markdown document using LangChain
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np

class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def get_embedding(self, text: str, mode: str = "sentence") -> np.ndarray:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        if mode == "sentence":
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif mode == "query":
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Неправильный режим. Используйте 'sentence' или 'query'")
            
        return embeddings.cpu().numpy()
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        batch_size = 32  # Увеличиваем размер батча для ускорения
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu().numpy())
            
            # Очищаем память
            del inputs, outputs
            if i % 100 == 0 and i > 0:
                torch.cuda.empty_cache()
                
        return embeddings
        
    def embed_query(self, text: str) -> np.ndarray:
        return self.get_embedding(text, mode="query")[0]  # Извлекаем первый элемент из батча

print("Загрузка документа и создание эмбеддингов...")
# Load your markdown file
loader = UnstructuredMarkdownLoader("./data/qa/mission_report.md")
docs = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

# Save chunks for later use
os.makedirs("qa_generated_data", exist_ok=True)
with open("qa_generated_data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print(f"Saved {len(chunks)} chunks to qa_generated_data/chunks.pkl")

embeddings = CustomHuggingFaceEmbeddings()

# Create a FAISS vector store from the document chunks and save it locally
vectorstore = FAISS.from_documents(chunks, embeddings)
# Сохраняем индекс в правильной директории, которую будет искать search_module.py
vectorstore.save_local("qa_generated_data/document_index")
print("Saved FAISS index to 'qa_generated_data/document_index'")

# Освобождаем память
del embeddings, vectorstore
gc.collect()
torch.cuda.empty_cache()

# ========= Part 2: QA Generation using Qwen Model =========

print("Загрузка модели для генерации вопросов...")
# Load the Qwen model with 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'  # Устанавливаем padding_side в 'left' для декодер-моделей
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Используем bfloat16 для ускорения
    quantization_config=quantization_config
)

def batch_generate(prompts: List[str], batch_size: int = 2) -> List[str]:
    """
    Given a list of prompt strings, returns a list of generated outputs.
    Processes prompts in small batches to avoid OOM.
    """
    if len(prompts) == 0:
        return []
        
    print(f"Генерация ответов для {len(prompts)} промптов маленькими батчами по {batch_size}...")
    
    def format_input(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True
        )
    
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        formatted = [format_input(p) for p in batch_prompts]
        
        # Токенизируем и обрезаем длинные промпты
        inputs = tokenizer(
            formatted, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=1536  # Увеличиваем максимальную длину входного текста
        ).to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=768,  # Увеличиваем для более полных ответов
                    temperature=0.2,  # Уменьшаем для более детерминированных ответов
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2  # Добавляем штраф за повторения
                )
                
                batch_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                results.extend(batch_outputs)
        except Exception as e:
            print(f"  Ошибка при генерации: {e}")
            # Добавляем пустые результаты при ошибке
            results.extend([""] * len(batch_prompts))
        finally:
            # Важно: сразу освобождаем память
            del inputs
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            
        # Показываем прогресс
        print(f"  Обработано {min(i + batch_size, len(prompts))}/{len(prompts)} промптов")
        
        # Добавляем небольшую паузу для стабилизации GPU
        time.sleep(0.5)
        
    return results

def parse_qa_block(block: str) -> Optional[Tuple[str, str, str]]:
    """
    Parses a QA block that should contain exactly three non-empty lines:
      - A line starting with "Question:"
      - A line starting with "Answer:"
      - A line starting with "Difficulty:"
    
    If the markers are not present but the block contains exactly three lines,
    those are used in order.
    
    Returns a tuple (question, answer, difficulty) or None if parsing fails.
    """
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    # Метод 1: Ищем строки с маркерами
    question, answer, difficulty = None, None, None
    for line in lines:
        lower = line.lower()
        if question is None and (lower.startswith("question:") or "question:" in lower):
            question = line.split(":", 1)[1].strip() if ":" in line else line
        elif answer is None and (lower.startswith("answer:") or "answer:" in lower):
            answer = line.split(":", 1)[1].strip() if ":" in line else line
        elif difficulty is None and (lower.startswith("difficulty:") or "difficulty:" in lower):
            difficulty = line.split(":", 1)[1].strip() if ":" in line else line

    if question and answer:
        # Проверка качества: должен быть настоящий вопрос, а не копия текста
        # Если вопрос содержит очевидные признаки копирования текста, пропускаем
        bad_indicators = ["==begin==", "==end==", "section page"]
        if any(ind in question.lower() for ind in bad_indicators):
            return None
            
        # Вопрос должен заканчиваться вопросительным знаком или быть формой вопроса
        question_indicators = ["?", "what", "who", "when", "where", "why", "how", "which"]
        is_proper_question = any(ind in question.lower() for ind in question_indicators)
        if not is_proper_question:
            return None
            
        # Если не нашли difficulty, установим medium по умолчанию
        return question, answer, difficulty or "medium"
        
    # Метод 2: Только если текст похож на вопрос-ответ
    if len(lines) >= 2 and lines[0].endswith("?"):
        question = lines[0]
        answer = lines[1]
        difficulty = lines[2] if len(lines) > 2 else "medium"
        return question, answer, difficulty
        
    return None

def parse_multiple_qa_output(output: str) -> List[Tuple[str, str, str]]:
    """
    Splits the output into blocks (separated by one or more blank lines) and
    attempts to parse each as a QA pair.
    
    Returns a list of successfully parsed QA tuples.
    """
    blocks = re.split(r'\n\s*\n', output.strip())
    qa_pairs = []
    for block in blocks:
        parsed = parse_qa_block(block)
        if parsed:
            qa_pairs.append(parsed)
    return qa_pairs

def generate_improved_prompt(context_before: str, context_current: str, context_after: str) -> str:
    """Создает улучшенный промпт для генерации вопросов"""
    # Ограничиваем длину контекстов для экономии токенов
    max_context_len = 600  
    context_before = context_before[-max_context_len:] if len(context_before) > max_context_len else context_before
    context_current = context_current[:max_context_len] if len(context_current) > max_context_len else context_current
    context_after = context_after[:max_context_len] if len(context_after) > max_context_len else context_after
    
    return f"""Вы - профессиональный ассистент, который создает вопросы и ответы для образовательных целей.

Текст: 
```
{context_before}

{context_current}

{context_after}
```

Инструкции:
1. Прочтите текст и создайте 2 вопроса с ответами по его содержанию.
2. Вопросы должны касаться фактов, упомянутых в ЦЕНТРАЛЬНОЙ части текста.
3. Вопросы должны заканчиваться вопросительным знаком.
4. Напишите также сложность вопроса: easy, medium или hard.
5. Используйте ТОЛЬКО следующий формат (строго 3 строки на каждый вопрос):

Question: [Ваш вопрос]
Answer: [Ответ на вопрос]
Difficulty: [Сложность]

Не добавляйте никаких комментариев или пояснений, только вопросы/ответы в указанном формате."""

def generate_question_batch_for_chunks(chunks: List, num_questions: int = 2, difficulty: str = None) -> List[Dict]:
    """
    Generates QA pairs for multiple chunks in batch.
    Обрабатывает промпты маленькими батчами для экономии памяти.
    Сохраняет результаты в один файл.
    """
    prompts = []
    chunk_ids = []
    
    print("Подготовка промптов...")
    # Prepare prompts using a sliding window
    for i in range(1, len(chunks) - 1):
        before = chunks[i-1].page_content
        current = chunks[i].page_content
        after = chunks[i+1].page_content
        
        # Используем улучшенный промпт
        prompt = generate_improved_prompt(before, current, after)
        prompts.append(prompt)
        chunk_ids.append(i)
    
    # Проверяем существует ли уже финальный файл с вопросами
    questions_path = os.path.join("qa_generated_data", "questions.json")
    if os.path.exists(questions_path):
        try:
            with open(questions_path, "r") as f:
                all_results = json.load(f)
                print(f"Загружен существующий файл с {len(all_results)} вопросами.")
                
                # Определяем, какие chunk_ids уже обработаны
                processed_chunks = set(item["chunk_id"] for item in all_results)
                
                # Фильтруем промпты, оставляя только необработанные
                new_prompts = []
                new_chunk_ids = []
                for i, (prompt, chunk_id) in enumerate(zip(prompts, chunk_ids)):
                    if chunk_id not in processed_chunks:
                        new_prompts.append(prompt)
                        new_chunk_ids.append(chunk_id)
                
                prompts = new_prompts
                chunk_ids = new_chunk_ids
                
                print(f"Осталось обработать {len(prompts)} промптов.")
                
                if not prompts:
                    print("Все промпты уже обработаны.")
                    return all_results
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при чтении {questions_path}: {e}")
            all_results = []
    else:
        all_results = []
    
    # Разбиваем все промпты на группы для инкрементальной обработки
    batch_size = 2  # Очень маленький батч для экономии памяти
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_chunk_ids = chunk_ids[start_idx:end_idx]
        
        print(f"\nОбработка батча {batch_idx+1}/{total_batches} ({start_idx+1}-{end_idx} из {len(prompts)})")
        
        # Пробуем обработать батч до 3 раз в случае неудачи
        max_retries = 3
        batch_results = []
        for retry in range(max_retries):
            if retry > 0:
                print(f"  Повторная попытка {retry}/{max_retries-1}...")
                time.sleep(2)  # Более длинная пауза перед повторной попыткой
            
            # Генерируем ответы для текущего батча
            outputs = batch_generate(batch_prompts, batch_size=batch_size)
            
            # Парсим результаты
            failed_count = 0
            
            for idx, output in enumerate(outputs):
                qa_pairs = parse_multiple_qa_output(output)
                if not qa_pairs or len(qa_pairs) < 1:  # Принимаем хотя бы 1 пару вместо 2
                    failed_count += 1
                    print(f"  Не удалось обработать промпт {start_idx + idx + 1}")
                else:
                    for qa in qa_pairs[:num_questions]:
                        batch_results.append({
                            "chunk_id": batch_chunk_ids[idx],
                            "question": qa[0],
                            "answer": qa[1],
                            "difficulty": qa[2]
                        })
            
            # Если удалось получить хотя бы одну пару, прерываем цикл retry
            if batch_results:
                break
        
        # Добавляем результаты этого батча к общим результатам
        all_results.extend(batch_results)
        print(f"  Батч {batch_idx+1}: добавлено {len(batch_results)} QA пар, пропущено {failed_count} промптов")
        
        # Сохраняем все накопленные результаты после каждого батча
        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  Общие результаты сохранены в {questions_path} (всего {len(all_results)} QA пар)")
        
        # Очищаем временные результаты батча после сохранения
        del batch_results
        
        # Очищаем память после каждого батча
        gc.collect()
        torch.cuda.empty_cache()
        
        # Добавляем паузу для стабилизации GPU
        time.sleep(1)
    
    return all_results

def cleanup_batch_files():
    """Удаляет промежуточные файлы батчей после успешного завершения"""
    for filename in os.listdir("qa_generated_data"):
        if filename.startswith("questions_batch_") and filename.endswith(".json"):
            try:
                os.remove(os.path.join("qa_generated_data", filename))
                print(f"Удален промежуточный файл: {filename}")
            except OSError as e:
                print(f"Ошибка при удалении {filename}: {e}")

print("Начинаем генерацию вопросов...")
# Generate QA pairs using small batches
all_questions = generate_question_batch_for_chunks(chunks, num_questions=2, difficulty="medium")
print(f"Generated {len(all_questions)} QA pairs.")

# Удаляем промежуточные файлы
cleanup_batch_files()

# Перезаписываем финальный файл с правильной кодировкой
questions_path = os.path.join("qa_generated_data", "questions.json")
with open(questions_path, "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)
print(f"Saved questions to {questions_path} with proper encoding")

print("Готово!") 