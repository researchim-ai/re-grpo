"""
Скрипт для автоматической генерации данных для обучения и оценки моделей вопросов и ответов (QA).

Выполняет две основные задачи:
1.  **Обработка исходного документа и создание векторного индекса:**
    - Загружает текстовый документ (в формате Markdown).
    - Разбивает документ на более мелкие фрагменты (чанки).
    - Генерирует векторные представления (эмбеддинги) для каждого чанка с помощью
      `CustomHuggingFaceEmbeddings` (на основе sentence-transformers).
    - Создает и сохраняет локально индекс FAISS из этих эмбеддингов для быстрого поиска
      похожих чанков. Чанки также сохраняются отдельно.

2.  **Генерация пар "вопрос-ответ" (QA) с использованием языковой модели (Qwen):**
    - Для каждого чанка (или группы чанков, формирующих контекст) генерируются промпты,
      предлагающие модели создать вопросы различной сложности и ответы на них,
      основанные на предоставленном тексте.
    - Используется модель Qwen (например, Qwen2.5-3B-Instruct) с 4-битной квантизацией для генерации.
    - Генерация происходит батчами для эффективного использования ресурсов и предотвращения ошибок OOM.
    - Сгенерированные QA-пары парсятся, фильтруются по качеству и сохраняются в формате JSON
      (`qa_generated_data/questions.json`).
    - Предусмотрена обработка ошибок, повторные попытки и сохранение промежуточных результатов
      для возможности возобновления генерации.

Требования к окружению (примерные):
    `pip install langchain faiss-cpu transformers bitsandbytes accelerate sentence-transformers`

Основные сохраняемые артефакты:
    - `qa_generated_data/chunks.pkl`: Сериализованный список текстовых чанков документа.
    - `qa_generated_data/document_index/`: Директория с файлами индекса FAISS.
    - `qa_generated_data/questions.json`: JSON-файл со списком сгенерированных QA-пар.
                                         Каждая пара включает `question`, `answer`, `difficulty`, `context`.
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
    """
    Класс для генерации эмбеддингов с использованием моделей Hugging Face.
    Предназначен для векторизации как документов (чанков), так и поисковых запросов.

    Примечание: Этот класс имеет схожее название и назначение с классами
    в `embeddings.py` и `search_module.py`. Рассмотрите возможность их объединения.

    Атрибуты:
        model_name (str): Название или путь к модели Hugging Face.
        model (AutoModel): Загруженная модель.
        tokenizer (AutoTokenizer): Загруженный токенизатор.
        device (torch.device): Устройство (CPU/CUDA), на котором работает модель.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def get_embedding(self, text: str, mode: str = "sentence") -> np.ndarray:
        """
        Генерирует эмбеддинг для одного текста.

        Args:
            text (str): Входной текст.
            mode (str): Режим ("sentence" для документов, "query" для запросов).
                        "sentence" использует усреднение всех токенов.
                        "query" использует эмбеддинг [CLS] токена (первого токена).

        Returns:
            np.ndarray: Эмбеддинг в виде NumPy массива.
        """
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
        """
        Генерирует эмбеддинги для списка документов (текстов) батчами.
        Использует режим "sentence".

        Args:
            texts (List[str]): Список текстов.

        Returns:
            List[np.ndarray]: Список эмбеддингов.
        """
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
        """
        Генерирует эмбеддинг для одного поискового запроса.
        Использует режим "query".

        Args:
            text (str): Текст запроса.

        Returns:
            np.ndarray: Эмбеддинг запроса.
        """
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
    Генерирует текстовые ответы для списка промптов с использованием загруженной LLM.
    Обрабатывает промпты небольшими батчами для предотвращения ошибок нехватки памяти (OOM).
    Применяет шаблон чата к каждому промпту перед передачей в модель.

    Args:
        prompts (List[str]): Список текстовых промптов для модели.
        batch_size (int): Размер батча для обработки промптов.

    Returns:
        List[str]: Список сгенерированных моделью текстовых ответов.
                   В случае ошибки генерации для батча, для промптов из этого батча
                   будут возвращены пустые строки.
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
    Парсит один блок текста, который должен содержать вопрос, ответ и сложность.

    Ожидаемый формат (с маркерами):
        Question: Текст вопроса
        Answer: Текст ответа
        Difficulty: Уровень сложности (easy, medium, hard)

    Если маркеры отсутствуют, но блок содержит ровно три непустые строки,
    они интерпретируются как вопрос, ответ и сложность в этом порядке.
    Выполняет базовые проверки качества вопроса (не должен быть копией контекста,
    должен содержать вопросительные слова или знак вопроса).

    Args:
        block (str): Строка, содержащая один QA-блок.

    Returns:
        Optional[Tuple[str, str, str]]: Кортеж (question, answer, difficulty),
        если парсинг успешен, иначе None. Сложность по умолчанию "medium", если не найдена.
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
    Парсит строку, которая может содержать несколько QA-блоков, разделенных "===" или "---".
    Каждый блок обрабатывается функцией `parse_qa_block`.

    Args:
        output (str): Строка с одним или несколькими QA-блоками.

    Returns:
        List[Tuple[str, str, str]]: Список успешно распарсенных QA-кортежей (question, answer, difficulty).
    """
    # Разделители могут быть разными, учитываем несколько вариантов
    # Также удаляем пустые строки после разделения
    blocks = re.split(r'\n\s*\n', output.strip())
    qa_pairs = []
    for block in blocks:
        parsed = parse_qa_block(block)
        if parsed:
            qa_pairs.append(parsed)
    return qa_pairs

def generate_improved_prompt(context_before: str, context_current: str, context_after: str, num_questions: int, difficulty: Optional[str] = None) -> str:
    """
    Формирует детализированный промпт для языковой модели для генерации QA-пар.

    Промпт включает:
    - Контекст: текущий чанк, а также предыдущий и последующий чанки для более широкого понимания.
    - Инструкции по генерации заданного количества (`num_questions`) вопросов.
    - Опциональное указание желаемой сложности (`difficulty`) вопросов (easy, medium, hard).
    - Требования к формату вывода (Question:, Answer:, Difficulty:).
    - Примеры ожидаемого формата.

    Args:
        context_before (str): Текст чанка, предшествующего текущему.
        context_current (str): Текст текущего чанка, на основе которого генерируются вопросы.
        context_after (str): Текст чанка, следующего за текущим.
        num_questions (int): Желаемое количество QA-пар для генерации.
        difficulty (Optional[str]): Желаемая сложность вопросов (например, "easy", "medium", "hard").
                                   Если None, модель сама определяет сложность.

    Returns:
        str: Сформированный текстовый промпт.
    """
    difficulty_prompt = f"The difficulty of the questions should be: {difficulty}." if difficulty else "Vary the difficulty of the questions (easy, medium, hard)."
    
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
1. Прочтите текст и создайте {num_questions} вопроса с ответами по его содержанию.
2. Вопросы должны касаться фактов, упомянутых в ЦЕНТРАЛЬНОЙ части текста.
3. Вопросы должны заканчиваться вопросительным знаком.
4. Напишите также сложность вопроса: {difficulty_prompt}
5. Используйте ТОЛЬКО следующий формат (строго {num_questions} строк на каждый вопрос):

Question: [Ваш вопрос]
Answer: [Ответ на вопрос]
Difficulty: [Сложность]

Не добавляйте никаких комментариев или пояснений, только вопросы/ответы в указанном формате."""

def generate_question_batch_for_chunks(chunks: List, num_questions_per_chunk: int = 2, target_total_questions: Optional[int] = None) -> List[Dict]:
    """
    Основная функция для генерации набора QA-пар на основе списка чанков документа.

    Итерирует по чанкам, используя скользящее окно для формирования контекста.
    Для каждого контекста генерирует промпт с помощью `generate_improved_prompt`,
    запрашивая `num_questions_per_chunk` вопросов.
    Вызывает `batch_generate` для получения ответов от LLM и `parse_multiple_qa_output` для их парсинга.

    Сохраняет все QA-пары в один файл `questions.json`, дописывая в него, если он уже существует.

    Args:
        chunks (List[Document]): Список документов (чанков), полученных из LangChain.
        num_questions_per_chunk (int): Целевое количество вопросов для генерации на каждый чанк/контекст.
        target_total_questions (Optional[int]): Опциональное общее количество QA-пар, которое нужно сгенерировать.
                                              Генерация остановится, когда это число будет достигнуто или превышено.

    Returns:
        List[Dict]: Список всех успешно сгенерированных и распарсенных QA-пар.
                    Каждый элемент - словарь с ключами "question", "answer", "difficulty", "context".
    """
    all_qa_pairs_final = []
    output_dir = "qa_generated_data"
    final_questions_path = os.path.join(output_dir, "questions.json")

    # Пытаемся загрузить существующие данные из questions.json
    if os.path.exists(final_questions_path):
        try:
            with open(final_questions_path, "r", encoding="utf-8") as f:
                all_qa_pairs_final = json.load(f)
            print(f"Загружено {len(all_qa_pairs_final)} существующих QA пар из {final_questions_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка чтения файла {final_questions_path}: {e}. Начинаем с пустого списка.")
            all_qa_pairs_final = []
    
    print("Подготовка промптов...")
    for i in range(len(chunks)):
        current_chunk_doc = chunks[i]
        current_chunk_text = current_chunk_doc.page_content
        
        before_chunk_text = chunks[i-1].page_content if i > 0 else ""
        after_chunk_text = chunks[i+1].page_content if i < len(chunks) - 1 else ""
        
        # Пропускаем генерацию для чанков, которые уже могли быть обработаны (по chunk_id)
        # Это простая проверка, чтобы избежать дублирования, если скрипт перезапускался.
        # Для более надежной проверки потребовалась бы более сложная логика отслеживания.
        already_processed_chunk_ids = {item.get("chunk_id") for item in all_qa_pairs_final}
        if i in already_processed_chunk_ids:
            print(f"  Чанк {i} уже обработан (найден в {final_questions_path}). Пропуск.")
            continue

        prompt = generate_improved_prompt(before_chunk_text, current_chunk_text, after_chunk_text, num_questions_per_chunk, difficulty="medium")
        
        print(f"\nОбработка чанка {i+1}/{len(chunks)}")
        outputs = batch_generate([prompt]) # batch_generate ожидает список
        
        generated_for_this_chunk = 0
        newly_generated_qa_for_chunk = []
        for output_text in outputs:
            qa_from_output = parse_multiple_qa_output(output_text)
            if qa_from_output:
                for q_text, a_text, diff_text in qa_from_output:
                    if generated_for_this_chunk < num_questions_per_chunk:
                        qa_pair = {
                            "chunk_id": i,
                            "question": q_text,
                            "answer": a_text,
                            "difficulty": diff_text,
                            "context": current_chunk_text
                        }
                        newly_generated_qa_for_chunk.append(qa_pair)
                        generated_for_this_chunk += 1
                    else:
                        break # Достигли нужного числа вопросов для этого чанка
            if generated_for_this_chunk >= num_questions_per_chunk:
                break # Если уже набрали достаточно с одного вывода LLM

        if not newly_generated_qa_for_chunk:
            print(f"  Не удалось сгенерировать QA для чанка {i}")
        else:
            print(f"  Для чанка {i} сгенерировано {len(newly_generated_qa_for_chunk)} QA пар.")
            all_qa_pairs_final.extend(newly_generated_qa_for_chunk)

        # Сохраняем текущие результаты в questions.json после каждого чанка, если что-то было добавлено
        if newly_generated_qa_for_chunk: 
            try:
                with open(final_questions_path, "w", encoding="utf-8") as f:
                    json.dump(all_qa_pairs_final, f, indent=2, ensure_ascii=False)
                print(f"  Промежуточные результаты ({len(all_qa_pairs_final)} QA пар) сохранены в {final_questions_path}")
            except IOError as e:
                print(f"  Ошибка сохранения промежуточных результатов в {final_questions_path}: {e}")

        # Очищаем память
        del outputs, newly_generated_qa_for_chunk, prompt
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5) # Небольшая пауза

        if target_total_questions and len(all_qa_pairs_final) >= target_total_questions:
            print(f"Достигнуто целевое количество вопросов ({target_total_questions}). Завершение генерации.")
            break

    # Финальное сохранение всех QA пар в один файл
    try:
        with open(final_questions_path, "w", encoding="utf-8") as f:
            json.dump(all_qa_pairs_final, f, indent=2, ensure_ascii=False)
        print(f"\nВсего сгенерировано и сохранено {len(all_qa_pairs_final)} QA пар в {final_questions_path}")
    except IOError as e:
        print(f"\nОшибка сохранения финального файла {final_questions_path}: {e}")
        print(f"Содержимое all_qa_pairs_final будет выведено здесь, если оно не пустое:")
        if all_qa_pairs_final:
             for item in all_qa_pairs_final:
                 print(item) # Печатаем для возможности ручного сохранения

    return all_qa_pairs_final

def cleanup_batch_files():
    """
    Удаляет временные файлы `.json`, созданные `generate_question_batch_for_chunks`.
    Вызывается после успешного объединения всех временных файлов в итоговый `questions.json`.
    """
    # Эта функция больше не нужна, так как временные файлы не создаются.
    # Оставляем ее пустой или удаляем, если она больше нигде не используется.
    print("Функция cleanup_batch_files() больше не выполняет никаких действий.")
    pass


print("Начинаем генерацию вопросов...")
# Generate QA pairs using small batches
all_questions = generate_question_batch_for_chunks(chunks, num_questions_per_chunk=2, target_total_questions=None)
print(f"Generated {len(all_questions)} QA pairs.")

# Удаляем промежуточные файлы
# cleanup_batch_files() # Больше не нужно

# Перезаписываем финальный файл с правильной кодировкой - это теперь делается в generate_question_batch_for_chunks
# questions_path = os.path.join("qa_generated_data", "questions.json")
# with open(questions_path, "w", encoding="utf-8") as f:
# json.dump(all_questions, f, indent=2, ensure_ascii=False)
# print(f"Saved questions to {questions_path} with proper encoding")

print("Готово!") 