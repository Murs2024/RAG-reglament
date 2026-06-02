"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import re
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

# Соответствие имени файла (без .txt) и кода для фильтра по кодексу/типу документа
# Обновлено под ваши файлы в data/
FILE_TO_CODE = {
    # Иски и кодексы
    "Исковое в суд по ГПК": "ГПК РФ",
    "gpk_rf": "ГПК РФ",
    "ИСКОВОЕ ЗАЯВЛЕНИЕ ПО АПК РФ": "АПК РФ",
    "apk_rf": "АПК РФ",
    "gk_rf1": "ГК РФ",
    "gk_rf2": "ГК РФ",
    "gk_rf3": "ГК РФ",
    # Претензия
    "ПРЕТЕНЗИЯ": "претензия",
    # Старые имена (если вернёте шаблоны)
    "letter_template": "письмо",
    "claim_pretenziya": "претензия",
    "claim_gpk": "ГПК РФ",
    "claim_apk": "АПК РФ",
}
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _clean_text(self, text: str) -> str:
        """
        Препроцессинг: очистка текста, удаление мусора (схема RAG — этап «Препроцессинг»).
        Все документы из data/ проходят эту очистку перед разбиением на чанки.
        """
        if not text or not text.strip():
            return ""
        # Удаление HTML-тегов
        text = re.sub(r"<[^>]+>", " ", text)
        # Множественные пробелы/табы — в один пробел (переносы строк сохраняем для абзацев)
        text = re.sub(r"[ \t]+", " ", text)
        # Множественные переносы — в один абзац (\n\n)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()
        return text
    
    def _read_text_file(self, file_path) -> str:
        """
        Чтение текстового файла. Сначала UTF-8, при ошибке — Windows-1251 (cp1251).
        Решает проблему файлов, сохранённых в кодировке Windows.
        """
        path = Path(file_path) if not isinstance(file_path, Path) else file_path
        for encoding in ("utf-8", "cp1251", "utf-8-sig"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Не удалось прочитать файл {path}: неизвестная кодировка")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents_from_folder(self, folder_path: str):
        """
        Загрузка всех .txt файлов из папки с метаданными source и code для фильтрации.
        source = имя файла без расширения, code = кодекс/тип (см. FILE_TO_CODE).
        Если коллекция уже содержит документы, загрузка пропускается.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Папка {folder_path} не найдена")
        
        txt_files = sorted(folder.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"В папке {folder_path} нет .txt файлов")
        
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию. Для перезагрузки удалите папку chroma_db/")
            return
        
        all_documents = []
        all_embeddings = []
        all_ids = []
        all_metadatas = []
        global_idx = 0
        
        for filepath in txt_files:
            stem = filepath.stem
            code = FILE_TO_CODE.get(stem, stem)
            
            text = self._read_text_file(filepath)
            
            # Препроцессинг: очистка текста, затем разбиение на чанки (схема RAG)
            text = self._clean_text(text)
            chunks = self._chunk_text(text)
            print(f"  {filepath.name}: {len(chunks)} чанков (code={code})")
            
            for chunk in chunks:
                embedding = self._create_embedding(chunk)
                all_documents.append(chunk)
                all_embeddings.append(embedding)
                all_ids.append(f"doc_{global_idx}")
                all_metadatas.append({"source": stem, "code": code})
                global_idx += 1
                if global_idx % 10 == 0:
                    print(f"  Обработано {global_idx} чанков...")
        
        self.collection.add(
            documents=all_documents,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadatas
        )
        print(f"Загружено {global_idx} чанков из {len(txt_files)} файлов в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            metadata_filter: фильтр по метаданным ChromaDB, например {"code": "ГПК РФ"}
                            или {"source": "claim_pretenziya"}. None — без фильтра.
            
        Returns:
            список документов с метаданными (text, id, distance, source, code)
        """
        query_embedding = self._create_embedding(query)
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter
        
        results = self.collection.query(**kwargs)
        
        documents = []
        if results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                doc = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                }
                if results.get("metadatas") and results["metadatas"][0]:
                    doc["source"] = results["metadatas"][0][i].get("source")
                    doc["code"] = results["metadatas"][0][i].get("code")
                documents.append(doc)
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов из папки data/
    if os.path.isdir("data"):
        vector_store.load_documents_from_folder("data")
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

