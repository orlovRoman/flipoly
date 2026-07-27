# Script for deployment
echo "?? ��������� pre-flight checks (�����)..."
$env:PYTHONPATH="."
python -m pytest tests
if ($LASTEXITCODE -ne 0) {
    echo "? ����� �����! ������ �������."
    exit $LASTEXITCODE
}

echo "? ����� ��������. �������� ������..."
echo "?? ��������� ������� ������..."

echo "1?? �������� ��������� � GitHub..."
    echo "?  !  ."
    exit $LASTEXITCODE
}

echo "?  .  ..."
echo "??   ..."

echo "1??    GitHub..."
git push origin main
if ($LASTEXITCODE -ne 0) {
    echo "?     GitHub"
    exit $LASTEXITCODE
}

echo "2?? Подключение к серверу и деплой..."
ssh agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732 'cd flipoly && git pull origin main && docker compose stop execution_worker_paper execution_worker_live scheduler api && docker compose build && docker compose run --rm api alembic upgrade head && docker compose up -d api && sleep 10 && curl --fail http://localhost:8001/dashboard && docker compose up -d scheduler execution_worker_paper'
if ($LASTEXITCODE -ne 0) {
    echo "? Ошибка при деплое на сервере"
    exit $LASTEXITCODE
}

echo "?   !"
