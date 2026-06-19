"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(self, 
                 collection_name: str = "rag_collection",
                 cache_db_path: str = "rag_cache.db",
                 data_dir: str = "data",
                 model: str = "gpt-4o-mini",
                 top_k: Optional[int] = None):
        """
        Инициализация RAG pipeline.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_dir: папка с .txt документами
            model: модель OpenAI для генерации ответов
            top_k: сколько чанков передавать в LLM (из векторной БД берутся самые релевантные).
                   Не зависит от размера БД: всегда берём топ-k по сходству. По умолчанию из TOP_K в .env или 5.
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")
        
        self.model = model
        self.top_k = top_k if top_k is not None else int(os.getenv("TOP_K", "5"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Инициализация компонентов
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)
        
        # Загрузка документов, если коллекция пустая
        if self.vector_store.collection.count() == 0:
            data_path = Path(data_dir)
            txt_in_dir = list(data_path.glob("*.txt")) if data_path.exists() else []
            if not txt_in_dir:
                raise FileNotFoundError(f"В папке {data_dir} нет .txt файлов для загрузки")
            print(f"Загрузка документов из папки {data_dir}...")
            self.vector_store.load_documents_from_folder(data_dir)
        
        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)
        
        self.data_dir = Path(data_dir)
        print("RAG Pipeline инициализирован (API mode)")
    
    def _is_form_pretenziya_query(self, query: str) -> bool:
        """Проверка: запрос о форме/реквизитах/шаблоне претензии."""
        q = (query or "").lower()
        triggers = [
            "форма претензии", "реквизиты претензии", "шаблон претензии",
            "форма и реквизиты претензии", "форма документа претензия",
            "претензия форма", "претензия реквизиты", "претензия шаблон",
        ]
        return any(t in q for t in triggers)

    def _get_form_pretenziya_from_file(self) -> Optional[str]:
        """Читает форму претензии из файла в data_dir (дословно). Возвращает None, если файла нет."""
        if not self.data_dir.exists():
            return None
        # Ищем файл с претензией по имени
        for name in ("ПРЕТЕНЗИЯ.txt", "претензия.txt", "Претензия.txt"):
            path = self.data_dir / name
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except Exception:
                    return None
        for f in self.data_dir.glob("*.txt"):
            if "претензи" in f.stem.lower():
                try:
                    return f.read_text(encoding="utf-8").strip()
                except Exception:
                    return None
        return None

    def _is_form_iskovoe_query(self, query: str) -> bool:
        """Проверка: запрос о форме/реквизитах/шаблоне искового заявления."""
        q = (query or "").lower()
        triggers = [
            "форма искового", "реквизиты искового", "шаблон искового",
            "исковое заявление форма", "исковое заявление реквизиты",
            "исковое заявление шаблон", "форма искового заявления",
            "исковое заявление-форма", "исковое заявление — форма",
        ]
        return any(t in q for t in triggers)

    def _get_form_iskovoe_from_file(self, query: str) -> Optional[tuple]:
        """
        Читает форму искового заявления из файла в data_dir (дословно).
        Возвращает (answer_text, context_docs) или None.
        Учитывает ГПК/АПК по тексту запроса.
        """
        if not self.data_dir.exists():
            return None
        q = (query or "").lower()
        # Собираем все файлы с «исков» в имени
        candidates = []
        for f in self.data_dir.glob("*.txt"):
            stem = f.stem.lower()
            if "исков" in stem:
                try:
                    text = f.read_text(encoding="utf-8").strip()
                    candidates.append((f.stem, text, stem))
                except Exception:
                    continue
        if not candidates:
            return None
        # По запросу выбираем: только АПК, только ГПК или оба
        apk = [(name, text) for name, text, stem in candidates if "апк" in stem]
        gpk = [(name, text) for name, text, stem in candidates if "гпк" in stem]
        parts = []
        context_docs = []
        if "апк" in q or "арбитраж" in q:
            if apk:
                name, text = apk[0]
                parts.append(f"Исковое заявление (АПК РФ), источник: {name}\n\n{text}")
                context_docs.append({"source": name, "code": "АПК РФ", "text": text})
        elif "гпк" in q or "гражданск" in q:
            if gpk:
                name, text = gpk[0]
                parts.append(f"Исковое заявление (ГПК РФ), источник: {name}\n\n{text}")
                context_docs.append({"source": name, "code": "ГПК РФ", "text": text})
        else:
            # без указания — отдаём оба по порядку ГПК, затем АПК
            for name, text in gpk:
                parts.append(f"Исковое заявление (ГПК РФ), источник: {name}\n\n{text}")
                context_docs.append({"source": name, "code": "ГПК РФ", "text": text})
            for name, text in apk:
                parts.append(f"Исковое заявление (АПК РФ), источник: {name}\n\n{text}")
                context_docs.append({"source": name, "code": "АПК РФ", "text": text})
        if not parts:
            return None
        answer = ("\n\n---\n\n".join(parts)).strip()
        return (answer, context_docs)

    def _format_sources_from_chunks(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        Формирует блок «Источники» из метаданных чанков, переданных в LLM.
        Выводится в ответе бота кодом, чтобы источник всегда был указан.
        """
        if not context_docs:
            return ""
        lines = ["---", "Источники (по чанкам из векторной БД, на основе которых сформирован ответ):"]
        for i, doc in enumerate(context_docs, 1):
            source = doc.get("source", "—")
            code = doc.get("code", "")
            ref = f"документ: {source}"
            if code:
                ref += f", кодекс: {code}"
            lines.append(f"  {i}. {ref}")
        return "\n".join(lines)
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для LLM с контекстом.
        
        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища
            
        Returns:
            сформированный промпт
        """
        # Формирование контекста: каждый фрагмент с явной ссылкой на документ/кодекс
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get("source", "документ")
            code = doc.get("code", "")
            ref = f"Источник: {source}" + (f", кодекс: {code}" if code else "")
            context_parts.append(f"[Документ {i}, {ref}]\n{doc['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Создание промпта: требуем от LLM ссылку на документ/кодекс в ответе
        prompt = f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст (у каждого фрагмента указан источник — документ и/или кодекс):
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Обязательно укажи в ответе ссылку на источник: документ или кодекс (например: «согласно ГПК РФ», «по документу ПРЕТЕНЗИЯ», «источник: претензия»). Без ссылки на источник ответ неполный
- Если вопрос касается формы документа, реквизитов, шаблона, перечня пунктов или структуры (например: «форма претензии», «реквизиты», «что должно быть в документе») — приводи текст из контекста дословно, как в источнике. Не переписывай список своими словами и не сокращай пункты: воспроизводи форму/структуру из документа без изменений
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным; для форм и реквизитов — дословно по документу
- Отвечай на русском языке

Ответ:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Генерация ответа через OpenAI API.
        
        Args:
            prompt: промпт для модели
            
        Returns:
            сгенерированный ответ
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Низкая температура для более точных ответов
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
    
    def query(
        self,
        user_query: str,
        use_cache: bool = True,
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Основной метод для обработки запроса пользователя через API.
        
        Поток:
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище (с опциональным фильтром по code/source)
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM API
        5. Сохранение в кеш
        
        Args:
            user_query: запрос пользователя
            use_cache: использовать ли кеш
            metadata_filter: фильтр поиска, напр. {"code": "ГПК РФ"} или {"source": "ПРЕТЕНЗИЯ"}
            
        Returns:
            словарь с ответом и метаданными
        """
        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        if metadata_filter:
            print(f"Фильтр: {metadata_filter}")
        print(f"{'='*60}")
        
        # Шаг 1: Сначала проверка в кеше; если есть — возврат, в БД не идём
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)
            
            if cached_result:
                print("[+] Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at")
                }
            print("[-] В кеше нет → идём в векторную БД")
        
        # Запрос о форме/реквизитах претензии — возвращаем документ из файла дословно (нормальная форма)
        if self._is_form_pretenziya_query(user_query):
            raw_form = self._get_form_pretenziya_from_file()
            if raw_form:
                print("[*] Запрос о форме претензии — возврат документа из файла (дословно)")
                answer = "Форма и реквизиты претензии (источник: ПРЕТЕНЗИЯ):\n\n" + raw_form
                context_docs = [{"source": "ПРЕТЕНЗИЯ", "code": "претензия", "text": raw_form}]
                sources_block = self._format_sources_from_chunks(context_docs)
                answer = answer.rstrip() + "\n\n" + sources_block
                if use_cache:
                    self.cache.set(user_query, answer, [raw_form])
                return {
                    "query": user_query,
                    "answer": answer,
                    "from_cache": False,
                    "context_docs": context_docs,
                    "model": None,
                    "mode": "API"
                }
        
        # Запрос о форме искового заявления — возвращаем документ из файла дословно
        if self._is_form_iskovoe_query(user_query):
            form_result = self._get_form_iskovoe_from_file(user_query)
            if form_result:
                raw_answer, context_docs = form_result
                print("[*] Запрос о форме искового заявления — возврат документа из файла (дословно)")
                sources_block = self._format_sources_from_chunks(context_docs)
                answer = raw_answer.rstrip() + "\n\n" + sources_block
                if use_cache:
                    self.cache.set(user_query, answer, [d.get("text", "") for d in context_docs])
                return {
                    "query": user_query,
                    "answer": answer,
                    "from_cache": False,
                    "context_docs": context_docs,
                    "model": None,
                    "mode": "API"
                }
        
        # Шаг 2: Поиск в векторной БД (ChromaDB); только если не было в кеше
        print("[*] Поиск в векторном хранилище (БД)...")
        context_docs = self.vector_store.search(
            user_query, top_k=self.top_k, metadata_filter=metadata_filter
        )
        print(f"[+] Найдено {len(context_docs)} релевантных чанков в векторной БД")
        # В логах — источник: какой вектор (чанк) из какого документа/кодекса
        for i, doc in enumerate(context_docs, 1):
            chunk_id = doc.get("id", "—")
            source = doc.get("source", "—")
            code = doc.get("code", "")
            distance = doc.get("distance")
            dist_str = f", distance={distance:.4f}" if distance is not None else ""
            print(f"    {i}. id={chunk_id} | документ: {source} | кодекс: {code}{dist_str}")
        
        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)
        
        # Шаг 4: Генерация ответа через API (LLM формирует ответ из переданных чанков)
        print(f"[*] Генерация ответа через OpenAI API ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")
        
        # В ответ бота обязательно добавляем блок «Источники» из чанков, что послали в LLM (кодом, не на совести LLM)
        sources_block = self._format_sources_from_chunks(context_docs)
        answer = answer.rstrip() + "\n\n" + sources_block
        
        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [doc['text'] for doc in context_docs]
            self.cache.set(user_query, answer, context_for_cache)
            print("[+] Сохранено в кеш")
        
        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.
        
        Returns:
            словарь со статистикой
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "top_k": self.top_k,
            "mode": "API"
        }


if __name__ == "__main__":
    # Тестирование RAG pipeline в API режиме
    import sys
    
    try:
        pipeline = RAGPipeline()
        
        # Тестовые запросы
        test_queries = [
            "Что такое машинное обучение?",
            "Что такое RAG?",
            "Как работают трансформеры?"
        ]
        
        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")
        
        # Повторный запрос (должен быть из кеша)
        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")
        
        # Статистика
        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
        print(f"Режим: {stats['mode']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

