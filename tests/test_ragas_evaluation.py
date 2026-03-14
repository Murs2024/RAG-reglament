"""
Тесты для evaluate_ragas.py: загрузка вопросов, фильтр, полная оценка RAGAS.
Запуск из корня проекта: pytest tests/test_ragas_evaluation.py -v
С API-ключом и интеграционный тест: pytest tests/test_ragas_evaluation.py -v -m integration
"""

import os
import sys
from pathlib import Path

import pytest

# Корень проекта — родитель папки tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Загрузка .env до импорта evaluate_ragas
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def test_questions_file_exists():
    """Файл evaluation_questions.txt существует в корне проекта."""
    qfile = PROJECT_ROOT / "evaluation_questions.txt"
    assert qfile.exists(), f"Файл не найден: {qfile}"


def test_load_questions_returns_list():
    """load_questions(exit_on_error=False) возвращает список строк (если файл есть)."""
    from evaluate_ragas import load_questions, QUESTIONS_FILE
    if not QUESTIONS_FILE.exists():
        pytest.skip("evaluation_questions.txt отсутствует")
    questions = load_questions(exit_on_error=False)
    assert isinstance(questions, list)
    if questions:
        assert all(isinstance(q, str) for q in questions)
        assert all(len(q.strip()) > 0 for q in questions)


def test_load_questions_non_empty():
    """В файле вопросов есть хотя бы один вопрос."""
    from evaluate_ragas import load_questions, QUESTIONS_FILE
    if not QUESTIONS_FILE.exists():
        pytest.skip("evaluation_questions.txt отсутствует")
    questions = load_questions(exit_on_error=False)
    assert len(questions) >= 1, "Добавьте вопросы в evaluation_questions.txt (один на строку)"


def test_infer_filter():
    """_infer_filter возвращает ожидаемый кодекс по ключевым словам."""
    from evaluate_ragas import _infer_filter
    assert _infer_filter("иск по ГПК") == {"code": "ГПК РФ"}
    assert _infer_filter("арбитражный суд АПК") == {"code": "АПК РФ"}
    assert _infer_filter("претензия по договору") == {"code": "претензия"}
    assert _infer_filter("нужно составить письмо контрагенту") is None
    assert _infer_filter("что-то без ключевых слов") is None


@pytest.mark.integration
def test_prepare_dataset_structure():
    """prepare_dataset возвращает Dataset с полями question, answer, contexts, ground_truth."""
    from datasets import Dataset
    from evaluate_ragas import prepare_dataset, load_questions, _infer_filter
    from rag_pipeline import RAGPipeline

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY не задан")
    questions = load_questions(exit_on_error=False)
    if not questions:
        pytest.skip("Нет вопросов в evaluation_questions.txt")
    # Один вопрос для быстроты
    one_question = [questions[0]]
    pipeline = RAGPipeline(
        collection_name="api_rag_collection",
        cache_db_path="api_rag_cache.db",
        data_dir="data",
        data_file="data/docs.txt",
        model="gpt-4o-mini",
    )
    dataset = prepare_dataset(pipeline, one_question)
    assert isinstance(dataset, Dataset)
    assert "question" in dataset.column_names
    assert "answer" in dataset.column_names
    assert "contexts" in dataset.column_names
    assert "ground_truth" in dataset.column_names
    assert len(dataset) == 1
    assert len(dataset["answer"][0]) > 0
    assert len(dataset["contexts"][0]) >= 0


@pytest.mark.integration
def test_run_ragas_evaluation_completes():
    """Полная оценка RAGAS завершается без ошибки и возвращает True (при наличии вопросов)."""
    from evaluate_ragas import run_ragas_evaluation, load_questions
    from rag_pipeline import RAGPipeline

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY не задан — интеграционный тест пропущен")
    questions = load_questions(exit_on_error=False)
    if not questions:
        pytest.skip("Нет вопросов в evaluation_questions.txt")
    pipeline = RAGPipeline(
        collection_name="api_rag_collection",
        cache_db_path="api_rag_cache.db",
        data_dir="data",
        data_file="data/docs.txt",
        model="gpt-4o-mini",
    )
    result = run_ragas_evaluation(pipeline)
    assert result is True
