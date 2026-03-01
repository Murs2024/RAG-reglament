if (Test-Path .\venv\Scripts\Activate.ps1) {
    .\venv\Scripts\Activate.ps1
    Write-Host "✅ Виртуальное окружение активировано!" -ForegroundColor Green
}
else {
    Write-Host "❌ Ошибка: venv не найдено" -ForegroundColor Red
    Write-Host "Создайте командой: python -m venv venv1" -ForegroundColor Yellow
}
