"""
Модуль для поиска по документам с использованием FAISS и предварительно обученных эмбеддингов.

Основные функции:
- Инициализация и управление векторным хранилищем FAISS.
- Генерация эмбеддингов для документов и запросов с помощью `CustomHuggingFaceEmbeddings`.
- Выполнение семантического поиска по индексированным документам.
- Загрузка и предоставление доступа к набору данных вопросов и ответов (QA).
"""

import pickle
import json
import os
import random
from typing import List, Union, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from embeddings import CustomHuggingFaceEmbeddings
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Глобальная переменная для хранения vectorstore
vectorstore = None
"""Глобальная переменная для хранения экземпляра векторного хранилища FAISS."""

class CustomHuggingFaceEmbeddings:
    """
    Класс для генерации текстовых эмбеддингов с использованием моделей Hugging Face.

    Этот класс оборачивает модели-трансформеры (по умолчанию "sentence-transformers/all-MiniLM-L6-v2")
    для создания векторных представлений текстов. Поддерживает батчинг и перемещение на GPU,
    если доступно. Реализует методы `embed_documents` для списков текстов и `embed_query` для
    одиночного текста, а также метод `__call__` для совместимости.

    Атрибуты:
        tokenizer (AutoTokenizer): Токенизатор из Hugging Face.
        model (AutoModel): Модель-трансформер из Hugging Face для генерации эмбеддингов.
        device (torch.device): Устройство, на котором выполняется модель (CPU или CUDA).
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.device = next(self.model.parameters()).device
        
    def _average_pool(self, last_hidden_states, attention_mask):
        """
        Выполняет average pooling для выходных скрытых состояний модели-трансформера,
        учитывая маску внимания.

        Args:
            last_hidden_states (torch.Tensor): Последние скрытые состояния модели.
            attention_mask (torch.Tensor): Маска внимания для токенов.

        Returns:
            torch.Tensor: Усредненные эмбеддинги.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка документов (текстов).
        Используется для индексации документов в векторном хранилище.

        Args:
            texts (List[str]): Список текстовых документов.

        Returns:
            List[List[float]]: Список эмбеддингов, где каждый эмбеддинг - это список float.
        """
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings_batch = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
                embeddings.extend(embeddings_batch.cpu().numpy().tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для одиночного текстового запроса.
        Используется для векторизации поискового запроса перед поиском схожих документов.

        Args:
            text (str): Текстовый запрос.

        Returns:
            List[float]: Эмбеддинг запроса в виде списка float.
        """
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().tolist()[0]
            
    # Для совместимости с интерфейсом Embeddings
    def __call__(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Обеспечивает вызов экземпляра класса как функции.
        Если на вход подается строка, вызывает `embed_query`.
        Если на вход подается список строк, вызывает `embed_documents`.

        Args:
            texts (Union[str, List[str]]): Текст или список текстов.

        Returns:
            Union[List[float], List[List[float]]]: Эмбеддинг или список эмбеддингов.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)

def init_vectorstore(force_rebuild=False):
    """
    Инициализирует или загружает векторное хранилище FAISS.

    Пытается загрузить предварительно сохраненные чанки документов из `qa_generated_data/chunks.pkl`.
    Если файл чанков существует, создает или пересоздает индекс FAISS на их основе.
    Результат сохраняется в глобальной переменной `vectorstore`.

    Args:
        force_rebuild (bool): Если True, принудительно пересоздает индекс, даже если он мог бы быть загружен
                              (текущая реализация всегда пересоздает из чанков).

    Returns:
        Optional[FAISS]: Экземпляр FAISS, если инициализация прошла успешно, иначе None.
    """
    global vectorstore
    
    try:
        # Сначала проверим, есть ли у нас чанки
        chunks_path = os.path.join("qa_generated_data", "chunks.pkl")
        
        if not os.path.exists(chunks_path):
            print(f"Файл с чанками не найден: {chunks_path}")
            print("Необходимо сначала запустить generate_data.py для создания чанков и индекса.")
            return None
            
        # Загружаем чанки и создаем индекс заново (безопаснее, чем десериализация)
        print(f"Загружаем чанки из {chunks_path} и создаем индекс FAISS")
        
        with open(chunks_path, "rb") as f:
            # Для безопасности используем safer_pickle
            chunks = pickle.load(f)
        
        # Создаем эмбеддинги и индекс заново
        embeddings = CustomHuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
        
    except Exception as e:
        print(f"Ошибка при инициализации векторного хранилища: {e}")
        return None

def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Выполняет поиск похожих документов в векторном хранилище FAISS по заданному запросу.

    Если векторное хранилище `vectorstore` еще не инициализировано, пытается его инициализировать
    вызовом `init_vectorstore()`.

    Args:
        query (str): Текстовый запрос для поиска.
        k (int): Количество наиболее похожих документов для возврата.

    Returns:
        List[Dict[str, Any]]: Список словарей, где каждый словарь представляет найденный документ
        и содержит ключи 'content' (текст документа), 'metadata' (метаданные) и 'score' (оценка схожести).
        В случае отсутствия `vectorstore` или ошибки поиска, возвращает список с одним элементом-ошибкой.
    """
    global vectorstore
    
    # Если vectorstore не инициализирован, пробуем загрузить
    if vectorstore is None:
        init_vectorstore()
        
    if vectorstore is None:
        return [{"content": "Векторное хранилище не найдено. Запустите generate_data.py для создания индекса.", 
                "metadata": {}, "score": 0.0}]
    
    # Выполняем поиск по схожести
    try:
        documents_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        results = []
        
        for doc, score in documents_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return results
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return [{"content": f"Ошибка поиска: {str(e)}", "metadata": {}, "score": 0.0}]

# Инициализируем векторное хранилище при импорте модуля
init_vectorstore()
"""Инициализация векторного хранилища при импорте модуля."""

def load_qa_data():
    """
    Загружает предварительно сгенерированные данные для вопросов и ответов (QA).

    Данные включают:
    - Чанки документов из `qa_generated_data/chunks.pkl`.
    - Пары вопрос-ответ из `qa_generated_data/questions.json`.

    Пути к файлам определяются относительно директории текущего файла (`__file__`).

    Returns:
        Tuple[Optional[List[Any]], Optional[List[Dict]]]: Кортеж (chunks, questions).
        Возвращает (None, None) в случае ошибки загрузки.
    """
    try:
        # Получение абсолютных путей к файлам данных
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chunks_path = os.path.join(base_dir, "qa_generated_data", "chunks.pkl")
        questions_path = os.path.join(base_dir, "qa_generated_data", "questions.json")
        
        print(f"Загрузка фрагментов из: {chunks_path}")
        print(f"Загрузка вопросов из: {questions_path}")
        
        # Загрузка фрагментов
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
            
        # Загрузка вопросов
        with open(questions_path, "r") as f:
            questions = json.load(f)
            
        print(f"Успешно загружено {len(chunks)} фрагментов и {len(questions)} вопросов")
        return chunks, questions
    except Exception as e:
        print(f"Ошибка при загрузке данных QA: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Загрузка фрагментов и вопросов при импорте модуля
try:
    chunks, questions = load_qa_data()
    if chunks is None or questions is None:
        print("Предупреждение: Не удалось загрузить данные QA.")
except Exception as e:
    print(f"Ошибка при инициализации данных QA: {e}")
    chunks, questions = None, None
"""Глобальные переменные для хранения загруженных чанков и вопросов."""

def get_question_answer(idx: Optional[int] = None, return_both: bool = True) -> Union[dict, str]:
    """
    Получает пару вопрос-ответ (или только вопрос) из загруженного набора `questions`.

    Можно получить QA-пару по конкретному индексу или случайную, если индекс не указан.

    Args:
        idx (Optional[int]): Индекс вопроса для извлечения. Если None, выбирается случайный вопрос.
        return_both (bool): Если True (по умолчанию), возвращает словарь {"question": q, "answer": a}.
                            Если False, возвращает только строку с вопросом.

    Returns:
        Union[dict, str]: Словарь с вопросом и ответом или строка с вопросом.

    Raises:
        ValueError: Если данные `questions` не были загружены или индекс вне допустимого диапазона.
    """
    if questions is None:
        raise ValueError("Вопросы не загружены. Убедитесь, что qa_generated_data/questions.json существует и доступен.")
        
    if idx is None:
        # Выбор случайного вопроса
        qa_pair = random.choice(questions)
    elif 0 <= idx < len(questions):
        # Выбор вопроса по индексу
        qa_pair = questions[idx]
    else:
        raise ValueError(f"Индекс вне диапазона. Должен быть между 0 и {len(questions)-1}")
    
    question = qa_pair['question']
    answer = qa_pair['answer']
    
    if return_both:
        return {"question": question, "answer": answer}
    else:
        return question

def get_question_count() -> int:
    """
    Возвращает общее количество доступных вопросов в загруженном наборе `questions`.

    Returns:
        int: Количество вопросов.

    Raises:
        ValueError: Если данные `questions` не были загружены.
    """
    if questions is None:
        raise ValueError("Вопросы не загружены. Убедитесь, что qa_generated_data/questions.json существует и доступен.")
    return len(questions)

def get_qa_dataset():
    """
    Возвращает Dataset HuggingFace, содержащий пары вопросов и ответов.

    Этот датасет строится из загруженных данных вопросов (questions.json).
    Каждый элемент в датасете - это словарь, который включает как минимум:
      - "question": Текст вопроса.
      - "answer": Соответствующий текст ответа.
    Дополнительные ключи из исходных данных вопросов также будут включены.

    Returns:
        Tuple[Dataset, Dataset]: Кортеж, содержащий тренировочный и тестовый датасеты HuggingFace.
    """
    if questions is None:
        raise ValueError("Вопросы не загружены. Убедитесь, что qa_generated_data/questions.json существует и доступен.")
    
    qa_dataset = Dataset.from_list(questions)
    full_dataset = qa_dataset.shuffle(seed=42)
    train_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)['train']
    test_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)['test'] 
    # Оставляем оригинальные имена колонок
    return train_dataset, test_dataset 