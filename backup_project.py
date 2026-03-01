"""
Скрипт для создания резервной копии проекта RAG-ассистент по регламентам и правовым документам.
Создает ZIP архив с файлами проекта. Архив сохраняется с датой и временем в папке backup.

Использование:
    python backup_project.py                      # Создать бэкап (без chroma_db)
    python backup_project.py --include-chroma     # Создать бэкап с векторной БД (chroma_db)
    python backup_project.py --list               # Показать список бэкапов
    python backup_project.py --cleanup            # Удалить старые, оставить 3

Что архивируется:
    Код (app.py, bot_telegram.py, rag_pipeline.py, cache.py, vector_store.py, evaluate_ragas.py и др.)
    Данные (data/), конфигурация (requirements.txt, .gitignore), отчёты и документация
    Скриншоты (assets/), кеш API (api_rag_cache.db при наличии)
    Исключено: .env, *.log, venv/, chroma_db/ (если не --include-chroma), backup/, __pycache__, .git
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ARCHIVE_DIR = Path(__file__).parent / "backup"
PROJECT_ROOT = Path(__file__).parent

EXCLUDE_DIRS = {
    "backup",
    "venv",
    ".venv",
    "__pycache__",
    ".vs",
    ".git",
    "chroma_db",
}


def should_exclude_path(path: Path, root: Path, exclude_dirs: set) -> bool:
    rel_path = path.relative_to(root)
    for part in rel_path.parts:
        if part in exclude_dirs:
            return True
    return False


def create_backup(include_chroma: bool = False):
    ARCHIVE_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    readable_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    archive_name = f"RAG_reglament_{readable_date}.zip"
    archive_path = ARCHIVE_DIR / archive_name

    print("=" * 60)
    print("БЭКАП ПРОЕКТА RAG-АССИСТЕНТ")
    print("=" * 60)
    print(f"Папка архива: {ARCHIVE_DIR}")
    print(f"Архив: {archive_name}")
    print(f"Дата: {now.strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    files_added = 0
    files_skipped = 0
    total_size = 0
    exclude_dirs = EXCLUDE_DIRS.copy()
    if include_chroma:
        exclude_dirs.discard("chroma_db")

    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for root, dirs, files in os.walk(PROJECT_ROOT):
                root_path = Path(root)
                try:
                    rel_root = root_path.relative_to(PROJECT_ROOT)
                except ValueError:
                    continue
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file in files:
                    file_path = root_path / file
                    try:
                        rel_file_path = file_path.relative_to(PROJECT_ROOT)
                    except ValueError:
                        continue

                    if file == ".env":
                        files_skipped += 1
                        continue
                    if file.endswith(".log"):
                        files_skipped += 1
                        continue
                    if should_exclude_path(file_path, PROJECT_ROOT, exclude_dirs):
                        files_skipped += 1
                        continue

                    arcname = file_path.relative_to(PROJECT_ROOT)
                    try:
                        zipf.write(file_path, arcname)
                        files_added += 1
                        total_size += file_path.stat().st_size
                    except Exception as e:
                        files_skipped += 1
    except Exception as e:
        print(f"ОШИБКА при создании архива: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return False

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print("БЭКАП СОЗДАН")
    print("-" * 60)
    print(f"Файл: {archive_name}")
    print(f"Путь: {archive_path}")
    print(f"Размер архива: {archive_size_mb:.2f} МБ")
    print(f"Файлов добавлено: {files_added}")
    print(f"Файлов пропущено: {files_skipped}")
    if total_size > 0:
        src_mb = total_size / (1024 * 1024)
        print(f"Исходный размер: {src_mb:.2f} МБ")
        print(f"Сжатие: {(1 - archive_size_mb / src_mb) * 100:.1f}%")
    print("=" * 60)
    cleanup_old_backups(keep_count=3)
    return True


def list_backups():
    if not ARCHIVE_DIR.exists():
        print("Папка backup не найдена.")
        return
    backups = sorted(
        list(ARCHIVE_DIR.glob("*.zip")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not backups:
        print("Бэкапы не найдены.")
        return
    print("БЭКАПЫ (последние 3)")
    print("-" * 60)
    for i, backup in enumerate(backups[:3], 1):
        size_mb = backup.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        date_str = mtime.strftime("%d.%m.%Y %H:%M")
        label = "САМЫЙ СВЕЖИЙ" if i == 1 else f"#{i}"
        print(f"  [{label}] {backup.name}  {date_str}  {size_mb:.2f} МБ")
    print("-" * 60)


def cleanup_old_backups(keep_count: int = 3):
    if not ARCHIVE_DIR.exists():
        return
    backups = sorted(
        list(ARCHIVE_DIR.glob("*.zip")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(backups) <= keep_count:
        return
    for backup in backups[keep_count:]:
        try:
            backup.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Резервная копия проекта RAG-ассистент")
    parser.add_argument("--list", action="store_true", help="Показать список бэкапов")
    parser.add_argument("--cleanup", action="store_true", help="Удалить старые бэкапы (оставить 3)")
    parser.add_argument(
        "--include-chroma",
        action="store_true",
        help="Включить папку chroma_db (векторная БД) в архив",
    )
    args = parser.parse_args()

    if args.list:
        list_backups()
        sys.exit(0)
    if args.cleanup:
        print("Очистка старых бэкапов...")
        cleanup_old_backups(keep_count=3)
        list_backups()
        sys.exit(0)

    if args.include_chroma:
        print("Режим: с векторной БД (chroma_db)")
    else:
        print("Режим: без chroma_db (используйте --include-chroma для включения)")
    print()

    if create_backup(include_chroma=args.include_chroma):
        print()
        list_backups()
