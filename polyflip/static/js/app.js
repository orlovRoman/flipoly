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
    
    // Simple helper to prevent XSS in innerHTML
    const escapeHtml = (unsafe) => {
        return String(unsafe)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    };
    
    // 1. Fetch Summary
    async function loadSummary() {
        try {
            const res = await fetch(window.API_BASE + '/api/analytics/summary');
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
    let chartInstances = {};
    let chartDataStore = {};

    async function loadCharts() {
        try {
            const res = await fetch(window.API_BASE + '/api/analytics/probabilities');
            chartDataStore = await res.json();
            renderSelectedChart();
        } catch (e) {
            console.error("Failed to load charts", e);
        }
    }

    function renderSelectedChart() {
        const selectedAsset = document.getElementById('asset-selector').value;
        const assetData = chartDataStore[selectedAsset] || {};
        const color = selectedAsset === 'BTC' ? '#0072F5' : '#00D395';
        
        const chartConfigs = [
            { id: 'chart-time', key: 'time_left_min', xTitle: 'Оставшееся время (минуты)' },
            { id: 'chart-price', key: 'mid_price', xTitle: 'Цена токена (USD)' },
            { id: 'chart-spread', key: 'spread', xTitle: 'Спред (USD)' },
            { id: 'chart-volume', key: 'volume_5min', xTitle: 'Объем торгов за 5 мин (USDC)' },
            { id: 'chart-velocity', key: 'price_velocity', xTitle: 'Скорость изменения цены за 5 мин' },
            { id: 'chart-hour', key: 'hour_of_day', xTitle: 'Время суток (UTC)' }
        ];

        chartConfigs.forEach(cfg => {
            const featureData = assetData[cfg.key] || { labels: [], probabilities: [] };
            const labels = featureData.labels;
            const points = featureData.probabilities.map(p => p * 100);
            
            const canvas = document.getElementById(cfg.id);
            if (!canvas) return; // safety check
            const ctx = canvas.getContext('2d');
            
            let existingChart = Chart.getChart(cfg.id);
            if (existingChart) existingChart.destroy();
            
            chartInstances[cfg.id] = createChart(ctx, labels, points, `${selectedAsset} Флип %`, color, cfg.xTitle);
        });
    }

    document.getElementById('asset-selector').addEventListener('change', renderSelectedChart);

    function createChart(ctx, labels, data, labelText, color, xTitle) {
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
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { 
                            color: '#8F9BB3',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#8F9BB3', maxRotation: 45, minRotation: 45 },
                        title: {
                            display: true,
                            text: xTitle,
                            color: '#8F9BB3',
                            font: { size: 13 }
                        }
                    }
                }
            }
        });
    }

    // 3. Fetch and Populate Settings
    async function loadSettings() {
        try {
            const res = await fetch(window.API_BASE + '/api/settings', {
                headers: getHeaders()
            });
            const data = await res.json();
            
            if(data.TRADE_EXECUTION_TIME_SEC) document.getElementById('TRADE_EXECUTION_TIME_SEC').value = data.TRADE_EXECUTION_TIME_SEC;
            if(data.TRADE_BET_SIZE_USDC) document.getElementById('TRADE_BET_SIZE_USDC').value = data.TRADE_BET_SIZE_USDC;
            if(data.TRADE_NO_FLIP_THRESHOLD) document.getElementById('TRADE_NO_FLIP_THRESHOLD').value = data.TRADE_NO_FLIP_THRESHOLD;
            if(data.TRADE_FLIP_THRESHOLD) document.getElementById('TRADE_FLIP_THRESHOLD').value = data.TRADE_FLIP_THRESHOLD;
            if(data.TRADING_ENABLED) document.getElementById('TRADING_ENABLED').checked = (data.TRADING_ENABLED === 'true');
            
            if(data.ACTIVE_FEATURES) {
                const active = data.ACTIVE_FEATURES.split(',');
                document.querySelectorAll('#ml-features input[type="checkbox"]').forEach(cb => {
                    cb.checked = active.includes(cb.value);
                });
            }
        } catch (e) {
            console.error("Failed to load settings", e);
        }
    }

    // === Button Handlers ===

    // Train Model
    document.getElementById('btn-train').addEventListener('click', async () => {
        const btn = document.getElementById('btn-train');
        btn.innerText = "Обучение...";
        btn.disabled = true;
        
        try {
            const res = await fetch(window.API_BASE + '/api/analytics/train', {
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
        
        // Сбор фичей
        const activeFeatures = Array.from(document.querySelectorAll('#ml-features input[type="checkbox"]:checked'))
                                    .map(cb => cb.value).join(',');
                                    
        const settingsToSave = {
            'ACTIVE_FEATURES': activeFeatures,
            'TRADE_EXECUTION_TIME_SEC': document.getElementById('TRADE_EXECUTION_TIME_SEC').value,
            'TRADE_BET_SIZE_USDC': document.getElementById('TRADE_BET_SIZE_USDC').value,
            'TRADE_NO_FLIP_THRESHOLD': document.getElementById('TRADE_NO_FLIP_THRESHOLD').value,
            'TRADE_FLIP_THRESHOLD': document.getElementById('TRADE_FLIP_THRESHOLD').value,
            'TRADING_ENABLED': document.getElementById('TRADING_ENABLED').checked ? 'true' : 'false'
        };
        
        let allOk = true;
        
        for (const [key, val] of Object.entries(settingsToSave)) {
            try {
                const res = await fetch(window.API_BASE + `/api/settings/${key}`, {
                    method: 'PUT',
                    headers: getHeaders(),
                    body: JSON.stringify({ value: String(val) })
                });
                if(!res.ok) allOk = false;
            } catch(err) {
                allOk = false;
            }
        }
        
        if(allOk) {
            alert("Настройки успешно сохранены!");
        } else {
            alert("Ошибка при сохранении части настроек. Проверьте API Key.");
        }
    });

    // 4. Fetch Parser Status
    async function loadParserStatus() {
        try {
            const res = await fetch(window.API_BASE + '/api/dashboard/status', {
                headers: getHeaders()
            });
            const data = await res.json();
            
            // 4.1 Collector Card
            const collector = data.collector;
            if (collector) {
                const statusSpan = document.getElementById('cs-status');
                statusSpan.innerText = collector.status;
                statusSpan.style.color = collector.status === 'success' ? 'var(--poly-green)' : '#ff3366';
                
                document.getElementById('cs-run-at').innerText = new Date(collector.run_at).toLocaleString();
                document.getElementById('cs-duration').innerText = collector.duration_sec;
                document.getElementById('cs-found').innerText = collector.markets_found;
                document.getElementById('cs-saved').innerText = collector.markets_saved;
                
                if(collector.error_message) {
                    document.getElementById('cs-error').innerText = "Error: " + collector.error_message;
                    document.getElementById('cs-error').style.display = 'block';
                } else {
                    document.getElementById('cs-error').style.display = 'none';
                }
            } else {
                document.getElementById('cs-status').innerText = "No data yet";
            }
            
            // 4.2 Dataset Table
            const dtBody = document.querySelector('#dataset-table tbody');
            dtBody.innerHTML = '';
            for (const [asset, counts] of Object.entries(data.dataset_summary)) {
                dtBody.innerHTML += `
                    <tr>
                        <td><strong>${escapeHtml(asset)}</strong></td>
                        <td style="color: var(--poly-green)">${counts.RESOLVED}</td>
                        <td style="color: #FFB020">${counts.PENDING}</td>
                    </tr>
                `;
            }
            if(Object.keys(data.dataset_summary).length === 0) {
                dtBody.innerHTML = `<tr><td colspan="3">Нет собранных данных</td></tr>`;
            }
            
            // 4.3 Live Markets Table
            const ltBody = document.querySelector('#live-table tbody');
            ltBody.innerHTML = '';
            for (const lm of data.live_markets) {
                ltBody.innerHTML += `
                    <tr>
                        <td><strong>${escapeHtml(lm.asset)}</strong></td>
                        <td style="font-size: 0.8rem">${escapeHtml(lm.question)}</td>
                        <td>${lm.current_yes_price}</td>
                        <td>${lm.current_spread}</td>
                        <td>${lm.volume_5min}</td>
                        <td style="font-size: 0.8rem">${new Date(lm.end_time_est).toLocaleTimeString()}</td>
                    </tr>
                `;
            }
            if(data.live_markets.length === 0) {
                ltBody.innerHTML = `<tr><td colspan="6">Нет активных рынков</td></tr>`;
            }

        } catch (e) {
            console.error("Failed to load parser status", e);
        }
    }

    // Refresh Status
    document.getElementById('btn-refresh-status').addEventListener('click', () => {
        loadSummary();
        loadCharts();
        loadParserStatus();
    });

    // === Init ===
    loadSummary();
    loadCharts();
    loadSettings();
    loadParserStatus();
});
