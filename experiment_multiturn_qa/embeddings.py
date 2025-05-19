"""
Модуль для работы с эмбеддингами с использованием Hugging Face моделей.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import numpy as np

class CustomHuggingFaceEmbeddings:
    """
    Класс для генерации эмбеддингов с использованием моделей Hugging Face.
    Поддерживает два режима: "sentence" для эмбеддингов документов и "query" для эмбеддингов запросов.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Инициализация модели и токенизатора.
        
        Args:
            model_name: Название модели из Hugging Face Hub
        """
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Использование GPU, если доступно
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def get_embedding(self, text: str, mode: str = "sentence") -> np.ndarray:
        """
        Получение эмбеддинга для текста.
        
        Args:
            text: Входной текст
            mode: Режим работы ("sentence" или "query")
            
        Returns:
            Эмбеддинг в виде numpy массива
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
        Получение эмбеддингов для списка документов.
        
        Args:
            texts: Список текстов документов
            
        Returns:
            Список эмбеддингов в виде numpy массивов
        """
        return [self.get_embedding(text, mode="sentence") for text in texts]
        
    def embed_query(self, text: str) -> np.ndarray:
        """
        Получение эмбеддинга для запроса.
        
        Args:
            text: Текст запроса
            
        Returns:
            Эмбеддинг в виде numpy массива
        """
        return self.get_embedding(text, mode="query") 