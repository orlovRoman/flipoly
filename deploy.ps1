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
git push origin main
if ($LASTEXITCODE -ne 0) {
    echo "? ������ ��� �������� � GitHub"
    exit $LASTEXITCODE
}

echo "2?? ����������� � ������� � ������..."
ssh agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732 'cd flipoly && git pull origin main && docker compose stop scheduler api && docker compose build && docker compose run --rm api alembic upgrade head && docker compose up -d api && curl --fail http://localhost:8001/health && docker compose up -d scheduler'
if ($LASTEXITCODE -ne 0) {
    echo "? ������ ��� ������ �� ������"
    exit $LASTEXITCODE
}

echo "? ������ ������� ��������!"
