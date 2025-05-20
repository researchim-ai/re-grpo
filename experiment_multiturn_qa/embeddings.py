"""
Модуль для работы с эмбеддингами с использованием Hugging Face моделей.

Содержит класс `CustomHuggingFaceEmbeddings` для генерации векторных представлений текста.

Примечание: Похожий по функциональности класс `CustomHuggingFaceEmbeddings` также существует
в модуле `search_module.py`. Следует рассмотреть возможность их объединения или устранения
избыточности, если их назначение полностью совпадает.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import numpy as np

class CustomHuggingFaceEmbeddings:
    """
    Класс для генерации эмбеддингов текста с использованием моделей Hugging Face.

    Позволяет получать векторные представления для отдельных текстов или списков текстов.
    Различает режимы для эмбеддинга документов ("sentence") и поисковых запросов ("query"),
    используя разные стратегии агрегации токеновых эмбеддингов.

    Атрибуты:
        model_name (str): Название или путь к модели Hugging Face.
        model (AutoModel): Загруженная модель Hugging Face.
        tokenizer (AutoTokenizer): Загруженный токенизатор Hugging Face.
        device (torch.device): Устройство, на котором выполняется модель (CPU или CUDA).

    Примечание: Этот класс имеет схожее название и назначение с классом в `search_module.py`.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Инициализирует модель и токенизатор Hugging Face.
        Модель автоматически перемещается на GPU, если он доступен.
        
        Args:
            model_name (str): Название или путь к предварительно обученной модели
                              из Hugging Face Hub (например, "sentence-transformers/all-mpnet-base-v2").
        """
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Использование GPU, если доступно
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def get_embedding(self, text: str, mode: str = "sentence") -> np.ndarray:
        """
        Генерирует эмбеддинг для одного входного текста.

        В зависимости от параметра `mode`:
        - "sentence": эмбеддинг вычисляется как среднее значение скрытых состояний всех токенов.
                      Предпочтительно для получения эмбеддинга документа.
        - "query": эмбеддинг берется из скрытого состояния первого токена (обычно [CLS] токена).
                     Предпочтительно для получения эмбеддинга поискового запроса.
        
        Args:
            text (str): Входной текст для генерации эмбеддинга.
            mode (str): Режим генерации эмбеддинга. Допустимые значения: "sentence" или "query".
            
        Returns:
            np.ndarray: Эмбеддинг текста в виде NumPy массива.

        Raises:
            ValueError: Если указан недопустимый `mode`.
        """
        # Токенизация текста
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Получение эмбеддингов
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Извлечение эмбеддингов
        if mode == "sentence":
            # Для документов используем среднее значение по всем токенам
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif mode == "query":
            # Для запросов используем эмбеддинг первого токена [CLS]
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Неправильный режим. Используйте 'sentence' или 'query'")
            
        # Преобразование в numpy массив
        return embeddings.cpu().numpy()
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Генерирует эмбеддинги для списка документов.
        Использует режим "sentence" для каждого документа.
        
        Args:
            texts (List[str]): Список текстов документов.
            
        Returns:
            List[np.ndarray]: Список эмбеддингов, где каждый эмбеддинг - это NumPy массив.
        """
        return [self.get_embedding(text, mode="sentence") for text in texts]
        
    def embed_query(self, text: str) -> np.ndarray:
        """
        Генерирует эмбеддинг для поискового запроса.
        Использует режим "query".
        
        Args:
            text (str): Текст поискового запроса.
            
        Returns:
            np.ndarray: Эмбеддинг запроса в виде NumPy массива.
        """
        return self.get_embedding(text, mode="query") 