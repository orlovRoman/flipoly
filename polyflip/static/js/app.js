document.addEventListener('DOMContentLoaded', () => {
    // === Tab Switching Logic ===
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active class from all
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));

            // Add active class to clicked
            item.classList.add('active');
            const targetId = item.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // === API Key Management ===
    // Simple helper to get headers
    function getHeaders() {
        const key = document.getElementById('api_key').value || 'test-key';
        return {
            'Content-Type': 'application/json',
            'X-API-Key': key
        };
    }

    // === Data Fetching & Rendering ===
    
    // 1. Fetch Summary
    async function loadSummary() {
        try {
            const res = await fetch('/api/analytics/summary');
            const data = await res.json();
            
            document.getElementById('stat-markets').innerText = data.total_resolved_markets || 0;
            document.getElementById('stat-flips').innerText = (data.flip_percentage || 0) + '%';
            
            const btcModel = data.active_models['BTC'];
            if(btcModel) {
                document.getElementById('stat-model-btc').innerText = `v${btcModel.version} (Acc: ${btcModel.accuracy})`;
            }
            
            const ethModel = data.active_models['ETH'];
            if(ethModel) {
                document.getElementById('stat-model-eth').innerText = `v${ethModel.version} (Acc: ${ethModel.accuracy})`;
            }
        } catch(e) {
            console.error("Failed to load summary", e);
        }
    }

    // 2. Fetch Probabilities and Render Charts
    let probChartInstance = null;
    let chartDataStore = {};

    async function loadCharts() {
        try {
            const res = await fetch('/api/analytics/probabilities');
            chartDataStore = await res.json();
            renderSelectedChart();
        } catch (e) {
            console.error("Failed to load charts", e);
        }
    }

    function renderSelectedChart() {
        const selectedAsset = document.getElementById('asset-selector').value;
        const assetData = chartDataStore[selectedAsset] || {};
        
        // X-axis: 0 to 15 minutes
        const labels = Array.from({length: 16}, (_, i) => i.toString());
        const points = labels.map(l => assetData[l] || 0);

        const color = selectedAsset === 'BTC' ? '#0072F5' : '#00D395';
        const ctx = document.getElementById('probChart').getContext('2d');
        
        if(probChartInstance) probChartInstance.destroy();
        probChartInstance = createChart(ctx, labels, points, `${selectedAsset} Вероятность Флипа`, color);
    }

    document.getElementById('asset-selector').addEventListener('change', renderSelectedChart);

    function createChart(ctx, labels, data, labelText, color) {
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: labelText,
                    data: data,
                    borderColor: color,
                    backgroundColor: color + '33', // 20% opacity
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4, // Smooth curves
                    pointBackgroundColor: color,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#8F9BB3' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#8F9BB3' },
                        title: {
                            display: true,
                            text: 'Осталось минут до закрытия',
                            color: '#8F9BB3'
                        }
                    }
                }
            }
        });
    }

    // === Button Handlers ===

    // Train Model
    document.getElementById('btn-train').addEventListener('click', async () => {
        const btn = document.getElementById('btn-train');
        btn.innerText = "Обучение...";
        btn.disabled = true;
        
        try {
            const res = await fetch('/api/analytics/train', {
                method: 'POST',
                headers: getHeaders()
            });
            if(res.ok) {
                alert("Задание на обучение отправлено в фон!");
            } else {
                alert("Ошибка запуска. Проверьте API Key.");
            }
        } catch(e) {
            console.error(e);
            alert("Network error.");
        } finally {
            btn.innerText = "Запустить переобучение";
            btn.disabled = false;
        }
    });

    // Save Settings
    document.getElementById('btn-save-settings').addEventListener('click', async (e) => {
        e.preventDefault();
        const interval = document.getElementById('poll_interval').value;
        
        try {
            const res = await fetch('/api/settings/LIVE_POLL_INTERVAL_SECONDS', {
                method: 'PUT',
                headers: getHeaders(),
                body: JSON.stringify({ value: interval })
            });
            
            if(res.ok) {
                alert("Настройки успешно сохранены!");
            } else {
                alert("Ошибка сохранения. Проверьте API Key.");
            }
        } catch(e) {
            console.error(e);
        }
    });

    // Refresh Status
    document.getElementById('btn-refresh-status').addEventListener('click', () => {
        // Just reload everything for now
        loadSummary();
        loadCharts();
    });

    // === Init ===
    loadSummary();
    loadCharts();
});
