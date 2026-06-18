"""
Telegram-бот для RAG-ассистента. Запуск: python bot_telegram.py
В Telegram задаёте вопрос — бот отвечает на основе регламентов и правовых документов.
"""

import os
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Загрузка .env (как в app.py)
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# --- Логирование ---
LOG_DIR = Path(__file__).resolve().parent
LOG_FILE = LOG_DIR / "bot_telegram.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging():
    """Логирование: консоль — только важное (INFO), файл — подробно (DEBUG)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Консоль — только INFO и выше (без мусора от telegram/httpx/httpcore)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Файл — подробно для отладки
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Сторонние библиотеки — в консоль не лезть, в файле только предупреждения
    for name in ("telegram", "telegram.ext", "httpx", "httpcore", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = None  # инициализируется в main()

# Фильтр по ключевым словам (как в app.py)
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
    q = (query or "").lower().strip()
    for keyword, code in QUERY_TO_FILTER.items():
        if keyword.lower() in q:
            return {"code": code}
    return None


# Глобальный пайплайн (инициализируется при старте)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if logger:
            logger.info("Инициализация RAG pipeline...")
        from rag_pipeline import RAGPipeline
        _pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_dir="data",
            model="gpt-4o-mini",
        )
        if logger:
            logger.info("RAG pipeline инициализирован")
    return _pipeline


TELEGRAM_MAX_MESSAGE = 4096  # лимит Telegram на одно сообщение


def _format_reply(result: dict) -> str:
    """Формирует текст ответа для Telegram (длинные ответы отправляются частями в handle_message)."""
    lines = [result["answer"]]
    if result.get("from_cache"):
        lines.append("\n💾 Из кеша.")
    if result.get("context_docs"):
        lines.append("\n📚 Источники:")
        for i, doc in enumerate(result["context_docs"][:5], 1):
            source = doc.get("source", "—")
            code = doc.get("code", "")
            ref = f"{code}, {source}" if code else source
            lines.append(f"  {i}. {ref}")
    return "\n".join(lines)


def _split_long_text(text: str, max_len: int = TELEGRAM_MAX_MESSAGE - 6) -> list:
    """Разбивает текст на части не длиннее max_len, по возможности по переводам строк."""
    if len(text) <= max_len:
        return [text] if text else []
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        block = text[:max_len]
        last_nl = block.rfind("\n")
        if last_nl > max_len // 2:
            cut = last_nl + 1
        else:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


async def handle_message(update, context):
    """Обработка текстового сообщения: вопрос -> RAG -> ответ в чат."""
    if not update.message or not update.message.text:
        if logger:
            logger.debug("handle_message: пустое сообщение или без текста, пропуск")
        return
    question = update.message.text.strip()
    user = update.effective_user
    user_id = user.id if user else None
    username = (user.username or user.first_name or "?") if user else "?"
    chat_id = update.effective_chat.id if update.effective_chat else None
    if logger:
        logger.info("Входящее сообщение | user_id=%s username=%s chat_id=%s text=%r", user_id, username, chat_id, question[:200])
    if not question:
        await update.message.reply_text("Напишите вопрос по регламентам или правовым документам.")
        if logger:
            logger.debug("Ответ: пустой вопрос — отправлена подсказка")
        return

    await update.message.reply_chat_action("typing")
    meta = _infer_filter(question)
    if logger:
        logger.debug("Фильтр по вопросу: %s", meta)

    loop = asyncio.get_event_loop()
    pipeline = get_pipeline()
    try:
        if logger:
            logger.info("Запрос в RAG: question=%r", question[:300])
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.query(question, metadata_filter=meta),
        )
        from_cache = result.get("from_cache", False)
        answer_len = len(result.get("answer", ""))
        ctx_count = len(result.get("context_docs") or [])
        if logger:
            logger.info("RAG ответ получен | from_cache=%s answer_len=%s context_docs=%s", from_cache, answer_len, ctx_count)
            logger.debug("Ответ (первые 500 символов): %s", (result.get("answer") or "")[:500])
        reply = _format_reply(result)
        chunks = _split_long_text(reply)
        for part in chunks:
            await update.message.reply_text(part)
        if logger:
            logger.info("Ответ отправлен в чат chat_id=%s reply_len=%s parts=%s", chat_id, len(reply), len(chunks))
    except Exception as e:
        if logger:
            logger.exception("Ошибка при обработке запроса: %s", e)
        await update.message.reply_text(f"Ошибка: {e}")


async def cmd_start(update, context):
    """Команда /start."""
    user = update.effective_user
    user_id = user.id if user else None
    username = (user.username or user.first_name or "?") if user else "?"
    chat_id = update.effective_chat.id if update.effective_chat else None
    if logger:
        logger.info("Команда /start | user_id=%s username=%s chat_id=%s", user_id, username, chat_id)
    await update.message.reply_text(
        "Привет. Я RAG-ассистент по регламентам и правовым документам.\n"
        "Задайте вопрос по ГПК, АПК, ГК РФ или претензиям — отвечу по загруженным документам.\n"
        "Подсказка: в вопросе укажите «ГПК», «АПК», «ГК РФ», «претензия» для поиска по нужному документу."
    )
    if logger:
        logger.debug("Ответ на /start отправлен")


async def cmd_help(update, context):
    """Команда /help."""
    if not update.message:
        return
    await update.message.reply_text(
        "📖 Помощь по боту\n\n"
        "• Напишите любой вопрос по регламентам или правовым документам — получу ответ по загруженным документам.\n"
        "• В вопросе можно указать: ГПК, АПК, ГК РФ, претензия — для поиска по нужному документу.\n"
        "• /start — приветствие\n"
        "• /stats — статистика (векторная БД, кеш, модель)\n"
        "• /clearcache или /clear — очистить кеш\n"
        "• /stop — выход из бота"
    )


async def cmd_stats(update, context):
    """Команда /stats — статистика RAG."""
    if not update.message:
        return
    await update.message.reply_chat_action("typing")
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        vs = stats.get("vector_store", {})
        cache = stats.get("cache", {})
        lines = [
            "📊 Статистика системы",
            "",
            "Векторное хранилище:",
            f"  документов (чанков): {vs.get('count', '—')}",
            "",
            "Кеш:",
            f"  записей: {cache.get('total_entries', '—')}",
            "",
            f"Модель: {stats.get('model', '—')}",
            f"top_k: {stats.get('top_k', '—')}",
        ]
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        if logger:
            logger.exception("Ошибка /stats: %s", e)
        await update.message.reply_text(f"Ошибка при получении статистики: {e}")


async def cmd_clearcache(update, context):
    """Команда /clearcache — очистка кеша RAG."""
    if not update.message:
        return
    try:
        pipeline = get_pipeline()
        pipeline.cache.clear()
        if logger:
            logger.info("Кеш очищен по команде /clearcache")
        await update.message.reply_text("✅ Кеш очищен. Следующие запросы будут заново обработаны через векторную БД и модель.")
    except Exception as e:
        if logger:
            logger.exception("Ошибка /clearcache: %s", e)
        await update.message.reply_text(f"Ошибка при очистке кеша: {e}")


async def cmd_stop(update, context):
    """Команда /stop — выход из бота (прощание)."""
    if not update.message:
        return
    await update.message.reply_text(
        "До свидания.\n"
        "Когда будете готовы снова — напишите /start."
    )


def main():
    global logger
    logger = setup_logging()
    logger.info("Запуск бота (bot_telegram.py) | log_file=%s", LOG_FILE)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан в .env")
        print("Ошибка: задайте TELEGRAM_BOT_TOKEN в .env")
        print("Создайте бота через @BotFather и вставьте токен в .env:")
        print("  TELEGRAM_BOT_TOKEN=123456:ABC-...")
        return

    logger.info("Токен бота загружен (длина=%s)", len(token))
    try:
        from telegram import Update, BotCommand  # type: ignore[reportMissingImports]
        from telegram.ext import Application, MessageHandler, CommandHandler, filters  # type: ignore[reportMissingImports]
        logger.debug("Импорт telegram и telegram.ext выполнен")
    except ImportError as e:
        logger.exception("Импорт telegram не удался: %s", e)
        print("Установите: pip install python-telegram-bot")
        return

    async def set_menu_commands(application: Application):
        """Установка меню команд в Telegram (кнопка рядом с полем ввода)."""
        await application.bot.set_my_commands([
            BotCommand("start", "Начать работу"),
            BotCommand("help", "Помощь по боту"),
            BotCommand("stats", "Статистика RAG"),
            BotCommand("clearcache", "Очистить кеш"),
            BotCommand("stop", "Выход из бота"),
        ])

    app = (
        Application.builder()
        .token(token)
        .post_init(set_menu_commands)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("clearcache", cmd_clearcache))
    app.add_handler(CommandHandler("clear", cmd_clearcache))  # алиас: /clear то же, что /clearcache
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Обработчики зарегистрированы: /start, /help, /stats, /clearcache, /clear, /stop, текст сообщений")

    logger.info("Polling запущен. Остановка: Ctrl+C")
    print("Бот запущен. Логи: консоль + %s" % LOG_FILE)
    print("Остановка: Ctrl+C")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
