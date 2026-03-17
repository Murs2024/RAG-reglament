"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
# Ищем .env в корне проекта (на уровень выше)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


# Файл с тестовыми вопросами (тот же, что для evaluate_ragas.py)
TEST_QUESTIONS_FILE = Path(__file__).parent / "evaluation_questions.txt"


def load_test_questions():
    """Вопросы для команды tests: из evaluation_questions.txt."""
    if not TEST_QUESTIONS_FILE.exists():
        return []
    lines = [line.strip() for line in TEST_QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines


# Ключевые слова в вопросе -> фильтр по полю code для поиска в базе
# Более длинные фразы выше, чтобы не перекрывались короткими (например «гражданский кодекс» до «гражданск»)
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
}


def _infer_filter(query: str):
    """По вопросу определить фильтр по кодексу/типу документа."""
    q = query.lower().strip()
    for keyword, code in QUERY_TO_FILTER.items():
        if keyword.lower() in q:
            return {"code": code}
    return None


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║   RAG-ассистент по регламентам и правовым документам    ║
║   Письмо | Претензия | Иск (ГПК РФ / АПК РФ)            ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Команды (латиница):")
    print("  exit, quit, q     — выход")
    print("  stats             — статистика (векторная БД, кеш, модель)")
    print("  clear             — очистка кеша (запросит подтверждение)")
    print("  tests             — показать вопросы из evaluation_questions.txt")
    print("  analyze           — разобрать последний ответ (RAGAS)")
    print("  eval              — оценить ответы по всем вопросам из файла (RAGAS)")
    print("Подсказка: в вопросе укажите «ГПК», «АПК», «ГК РФ», «претензия» — поиск по нужному документу.\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("💾 Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"🌐 Источник: OpenAI API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n💬 Ответ:\n{result['answer']}")
    
    # Показать контекст и ссылки на документ/кодекс
    if not result['from_cache'] and result.get('context_docs'):
        print(f"\n📚 Источники (документ / кодекс):")
        for i, doc in enumerate(result['context_docs'][:5], 1):
            source = doc.get("source", "—")
            code = doc.get("code", "")
            ref = f"{code}, {source}" if code else source
            preview = doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text']
            print(f"   {i}. [{ref}]")
            print(f"      {preview}")
    
    print(f"{'─'*60}\n")


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\n🤖 Модель: {stats['model']}")
    print(f"📄 Чанков в LLM (top_k): {stats.get('top_k', '—')}")
    print(f"🌐 Режим: {stats['mode']}")
    print(f"{'═'*60}\n")


def main():
    """Главная функция приложения."""
    print_banner()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    try:
        # Инициализация RAG pipeline
        print("🚀 Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_dir="data",
            data_file="data/docs.txt",
            model="gpt-4o-mini"
        )
        print("\n✅ Система готова к работе!\n")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    last_result = None  # для команды analyze (последний ответ в боте)
    
    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("💭 Ваш вопрос: ").strip()
            
            # Обработка специальных команд (только в консоли; латиница)
            cmd = user_input.lower().strip()
            if cmd in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if cmd == 'stats' or cmd == 'статистика':
                print_stats(pipeline)
                continue
            
            if cmd == 'tests':
                test_questions = load_test_questions()
                print(f"\n{'═'*60}")
                print("📋 ТЕСТОВЫЕ ВОПРОСЫ (из evaluation_questions.txt)")
                print(f"{'═'*60}\n")
                if not test_questions:
                    print("  Файл evaluation_questions.txt не найден или пуст.\n")
                else:
                    for i, q in enumerate(test_questions, 1):
                        print(f"  {i}. {q}")
                    print(f"\n  Скопируйте любой вопрос и вставьте в «Ваш вопрос».\n")
                continue
            
            if cmd == 'analyze':
                if last_result is None:
                    print("\n[!] Сначала задайте любой вопрос и получите ответ, затем введите «analyze».\n")
                else:
                    try:
                        from evaluate_ragas import run_ragas_single
                        q = last_result.get("query", "")
                        a = last_result.get("answer", "")
                        docs = last_result.get("context_docs") or []
                        ctx = [d.get("text", d) if isinstance(d, dict) else d for d in docs]
                        run_ragas_single(q, a, ctx)
                    except Exception as e:
                        print(f"\n[!] Ошибка: {e}\n")
                continue
            
            if cmd == 'eval':
                try:
                    from evaluate_ragas import run_ragas_evaluation
                    run_ragas_evaluation(pipeline)
                except Exception as e:
                    print(f"\n[!] Ошибка оценки RAGAS: {e}\n")
                continue
            
            if cmd == 'clear':
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("✅ Кеш очищен")
                continue
            
            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue
            
            # Определяем фильтр по кодексу/типу из формулировки вопроса
            metadata_filter = _infer_filter(user_input)
            
            # Обработка запроса через RAG pipeline
            result = pipeline.query(user_input, metadata_filter=metadata_filter)
            last_result = result  # для команды «analyze»
            
            # Вывод результата
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

