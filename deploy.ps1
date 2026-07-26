# Скрипт для деплоя
echo "🚀 Запускаем процесс деплоя..."

echo "1️⃣ Отправка изменений в GitHub..."
git push origin main
if ($LASTEXITCODE -ne 0) {
    echo "❌ Ошибка при отправке в GitHub"
    exit $LASTEXITCODE
}

echo "2️⃣ Подключение к серверу и деплой..."
ssh agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732 "cd flipoly && git pull origin main && docker compose up -d --build"
if ($LASTEXITCODE -ne 0) {
    echo "❌ Ошибка при деплое на сервер"
    exit $LASTEXITCODE
}

echo "✅ Деплой успешно завершен!"
