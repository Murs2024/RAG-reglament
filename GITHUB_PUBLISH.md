# Публикация проекта на GitHub (Murs2024)

Репозиторий: **https://github.com/Murs2024**

## Шаг 1. Создать новый репозиторий на GitHub

1. Откройте https://github.com/new (или **Repositories** → **New**).
2. **Repository name:** например `RAG-reglament` или `rag-assistant-reglament`.
3. **Description (необязательно):** RAG-ассистент по регламентам и правовым документам.
4. Выберите **Public**.
5. **Не** ставьте галочки "Add a README" / "Add .gitignore" — проект уже есть локально.
6. Нажмите **Create repository**.

## Шаг 2. В папке проекта выполнить команды

Откройте терминал в корне проекта (папка с `app.py`, `README.md`) и выполните по порядку:

```bash
# Инициализация репозитория
git init

# Добавить все файлы (исключения — по .gitignore: venv, chroma_db, .env, backup и т.д.)
git add .

# Первый коммит
git commit -m "Initial commit: RAG-ассистент по регламентам и правовым документам"

# Подключить ваш репозиторий (замените REPO_NAME на имя из шага 1, например RAG-reglament)
git remote add origin https://github.com/Murs2024/REPO_NAME.git

# Основная ветка (если GitHub просит main)
git branch -M main

# Отправить код на GitHub
git push -u origin main
```

**Важно:** замените `REPO_NAME` на фактическое имя репозитория (например `RAG-reglament`).

При `git push` браузер или Git могут запросить авторизацию (логин/пароль или токен). Для HTTPS используйте [Personal Access Token](https://github.com/settings/tokens) вместо пароля.

## Что не попадёт в репозиторий (уже в .gitignore)

- `venv/`, `chroma_db/`, `backup/`
- `.env` (секреты)
- `__pycache__/`, логи

После выполнения этих шагов проект будет доступен по адресу:  
**https://github.com/Murs2024/REPO_NAME**
