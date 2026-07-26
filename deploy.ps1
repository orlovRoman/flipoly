# Script for deployment
echo "🚀 Запускаем pre-flight checks (тесты)..."
$env:PYTHONPATH="."
python -m pytest tests
if ($LASTEXITCODE -ne 0) {
    echo "❌ Тесты упали! Деплой отменен."
    exit $LASTEXITCODE
}

echo "✅ Тесты пройдены. Начинаем деплой..."
echo "🚀 Запускаем процесс деплоя..."

echo "1️⃣ Отправка изменений в GitHub..."
git push origin main
if ($LASTEXITCODE -ne 0) {
    echo "❌ Ошибка при отправке в GitHub"
    exit $LASTEXITCODE
}

echo "2️⃣ Подключение к серверу и деплой..."
ssh agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732 'cd flipoly && git pull origin main && docker compose up -d --build && docker compose exec -T api alembic upgrade head'
if ($LASTEXITCODE -ne 0) {
    echo "❌ Ошибка при деплое на сервер"
    exit $LASTEXITCODE
}

echo "✅ Деплой успешно завершен!"
