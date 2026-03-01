"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate

# Правильный импорт для RAGAS 0.4.x - используем классы метрик
try:
    # Новый способ импорта (RAGAS 0.4+)
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision
    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        # Альтернативный импорт из collections
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        # Fallback на старый импорт
        from ragas.metrics import faithfulness, context_precision

from rag_pipeline import RAGPipeline


# Файл с вопросами для оценки: один вопрос на строку (в корне проекта)
QUESTIONS_FILE = Path(__file__).parent / "evaluation_questions.txt"

# Сколько вопросов использовать для оценки (первые N из файла)
MAX_QUESTIONS = 3

# Фильтр по ключевым словам (как в app.py) — для согласованной оценки с ботом
QUERY_TO_FILTER = {
    "гражданский кодекс": "ГК РФ",
    "ГК РФ": "ГК РФ",
    "гк рф": "ГК РФ",
    "ГПК": "ГПК РФ",
    "гпк": "ГПК РФ",
    "гражданск": "ГПК РФ",
    "мировой суд": "ГПК РФ",
    "районный суд": "ГПК РФ",
    "АПК": "АПК РФ",
    "апк": "АПК РФ",
    "арбитраж": "АПК РФ",
    "претензи": "претензия",
    "письм": "письмо",
}


def _infer_filter(query: str):
    """По вопросу определить фильтр по кодексу/типу документа (как в app.py)."""
    q = (query or "").lower().strip()
    for keyword, code in QUERY_TO_FILTER.items():
        if keyword.lower() in q:
            return {"code": code}
    return None


def load_questions(exit_on_error: bool = True):
    """
    Вопросы для оценки из файла evaluation_questions.txt.
    Args:
        exit_on_error: если True и файла нет/пустой — sys.exit(1). Если False — вернуть [].
    Returns:
        list of questions (или [] при ошибке, если exit_on_error=False)
    """
    if not QUESTIONS_FILE.exists():
        msg = f"Файл не найден: {QUESTIONS_FILE}. Создайте evaluation_questions.txt (один вопрос на строку, UTF-8)."
        if exit_on_error:
            print(f"[ОШИБКА] {msg}")
            sys.exit(1)
        return []
    lines = [line.strip() for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        msg = f"В файле {QUESTIONS_FILE.name} нет вопросов (пустые строки игнорируются)."
        if exit_on_error:
            print(f"[ОШИБКА] {msg}")
            sys.exit(1)
        return []
    return lines


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без кеша; фильтр по вопросу — как в боте)
        meta = _infer_filter(question)
        result = pipeline.query(question, use_cache=False, metadata_filter=meta)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def run_ragas_evaluation(pipeline):
    """
    Запуск оценки RAG через RAGAS для проверки ответов LLM.
    Можно вызывать из бота (app.py) по команде eval — передаётся текущий pipeline.
    
    Args:
        pipeline: экземпляр RAGPipeline (уже инициализированный).
    Returns:
        True если оценка выполнена, False если нет вопросов в файле.
    """
    import math
    questions = load_questions(exit_on_error=False)
    if not questions:
        print("\n[!] Нет вопросов для оценки. Добавьте их в evaluation_questions.txt (один на строку).\n")
        return False
    questions = questions[:MAX_QUESTIONS]

    print("\n" + "=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG (RAGAS)")
    print("=" * 70)
    print(f"[*] Вопросов из {QUESTIONS_FILE.name}: {len(questions)} шт. (используем до {MAX_QUESTIONS})\n")
    
    dataset = prepare_dataset(pipeline, questions)
    
    print("[*] Запуск метрик RAGAS (Faithfulness, Context Precision)...")
    metrics_to_use = [faithfulness(), context_precision()]
    try:
        result = evaluate(dataset=dataset, metrics=metrics_to_use)
    except Exception as e:
        print(f"[ОШИБКА] Оценка RAGAS: {e}\n")
        return False
    
    faithfulness_values = [v for v in result['faithfulness'] if not (isinstance(v, float) and math.isnan(v))]
    context_precision_values = [v for v in result['context_precision'] if not (isinstance(v, float) and math.isnan(v))]
    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_context_precision = sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    
    print("\n[МЕТРИКИ]")
    print(f"   Faithfulness (точность ответа):     {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста): {avg_context_precision:.4f}")
    print(f"   ИТОГО средний балл:                {avg_score:.4f}")
    if avg_score >= 0.7:
        print("   Оценка: отличное качество [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: удовлетворительно [!]")
    else:
        print("   Оценка: требуется улучшение [X]")
    print("=" * 70 + "\n")
    return True


def run_ragas_single(question: str, answer: str, contexts: list):
    """
    Анализ одного ответа (тот, что только что получили в боте).
    Команда в боте: сначала задайте вопрос, получите ответ, затем введите «analyze».
    
    Args:
        question: вопрос пользователя
        answer: ответ LLM
        contexts: список текстов чанков (контекст, который передавали в LLM)
    """
    import math
    if not contexts:
        print("\n[!] Нет контекста для анализа (ответ мог быть из кеша без сохранения контекста). Задайте вопрос заново.\n")
        return
    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [answer[:100]],
    }
    dataset = Dataset.from_dict(dataset_dict)
    metrics_to_use = [faithfulness(), context_precision()]
    try:
        result = evaluate(dataset=dataset, metrics=metrics_to_use)
    except Exception as e:
        print(f"\n[!] Ошибка RAGAS: {e}\n")
        return
    f_val = result["faithfulness"][0]
    cp_val = result["context_precision"][0]
    f_ok = not (isinstance(f_val, float) and math.isnan(f_val))
    cp_ok = not (isinstance(cp_val, float) and math.isnan(cp_val))
    print("\n" + "=" * 60)
    print("АНАЛИЗ ПОСЛЕДНЕГО ОТВЕТА (RAGAS)")
    print("=" * 60)
    print(f"Вопрос: {question[:80]}...")
    print(f"\n   Faithfulness (ответ по контексту):  {f_val:.4f}" if f_ok else "   Faithfulness: —")
    print(f"   Context Precision (релевантность):  {cp_val:.4f}" if cp_ok else "   Context Precision: —")
    print("=" * 60 + "\n")


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_dir="data",
            data_file="data/docs.txt",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Вопросы и запуск оценки (общая логика в run_ragas_evaluation)
    questions = load_questions()[:MAX_QUESTIONS]
    print("=" * 70)
    print(f"[*] Вопросы из файла {QUESTIONS_FILE.name}: {len(questions)} шт. (до {MAX_QUESTIONS})")
    print("=" * 70)
    dataset = prepare_dataset(pipeline, questions)
    print("=" * 70)
    print("\n[*] Запуск метрик RAGAS (1–2 мин)...\n")
    metrics_to_use = [faithfulness(), context_precision()]
    try:
        result = evaluate(dataset=dataset, metrics=metrics_to_use)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    import math
    faithfulness_values = [v for v in result['faithfulness'] if not (isinstance(v, float) and math.isnan(v))]
    context_precision_values = [v for v in result['context_precision'] if not (isinstance(v, float) and math.isnan(v))]
    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_context_precision = sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    for i, question in enumerate(questions):
        print(f"\n{i+1}. {question}")
        faith_val = result['faithfulness'][i]
        cp_val = result['context_precision'][i]
        print(f"   Faithfulness:       {faith_val:.4f}" if not (isinstance(faith_val, float) and math.isnan(faith_val)) else "   Faithfulness:       —")
        print(f"   Context Precision:  {cp_val:.4f}" if not (isinstance(cp_val, float) and math.isnan(cp_val)) else "   Context Precision:  —")
    print("\n" + "=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

