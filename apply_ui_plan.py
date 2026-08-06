import re
import os

path = "C:/Users/orlov/.gemini/antigravity/scratch/flipoly/polyflip/templates/execution.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update wrappers and panel IDs
# Panel 3.5 (Resolved)
content = content.replace(
    '<!-- Блок 3.5. Завершённые рынки -->\n            <div class="panel" style="margin-bottom: 20px;">',
    '<!-- Блок 3.5. Завершённые рынки -->\n            <div class="panel" id="resolved-panel" style="margin-bottom: 20px;">'
)
content = content.replace(
    '<h3>Завершённые рынки</h3>',
    '<h3>Завершённые рынки</h3>\n                    <button class="btn-action" style="background:#4a5568; padding:2px 8px; font-size:0.8rem;" onclick="togglePanel(\'resolved-panel\', this)">Свернуть</button>'
)

# Panel 3.6 (Archive)
content = content.replace(
    '<!-- Блок 3.6. Архив -->\n            <div class="panel" style="margin-bottom: 20px;">',
    '<!-- Блок 3.6. Архив -->\n            <div class="panel" id="archive-panel" style="margin-bottom: 20px;">'
)
content = content.replace(
    '<h3>Архив</h3>',
    '<h3>Архив</h3>\n                    <button class="btn-action" style="background:#4a5568; padding:2px 8px; font-size:0.8rem;" onclick="togglePanel(\'archive-panel\', this)">Свернуть</button>'
)

# Panel 4 (Requests)
content = content.replace(
    '<!-- Блок 4. Заявки -->\n            <div class="panel" style="margin-bottom: 20px;">',
    '<!-- Блок 4. Заявки -->\n            <div class="panel" id="requests-panel" style="margin-bottom: 20px;">'
)
content = content.replace(
    '<h3>LIVE-Заявки (Execution Requests)</h3>',
    '<h3>LIVE-Заявки (Execution Requests)</h3>\n                    <button class="btn-action" style="background:#4a5568; padding:2px 8px; font-size:0.8rem;" onclick="togglePanel(\'requests-panel\', this)">Свернуть</button>'
)

# Tradable Panel (Block 3)
content = content.replace(
    '<!-- Блок 3. Торгуемые позиции -->\n            <div class="panel" style="margin-bottom: 20px;">',
    '<!-- Блок 3. Торгуемые позиции -->\n            <div class="panel" id="tradable-panel" style="margin-bottom: 20px;">'
)
content = content.replace(
    '<h3>Торгуемые позиции</h3>',
    '<h3>Торгуемые позиции</h3>\n                    <button class="btn-action" style="background:#4a5568; padding:2px 8px; font-size:0.8rem;" onclick="togglePanel(\'tradable-panel\', this)">Свернуть</button>'
)

# Add table-wrapper class to the overflow-x divs that contain tables
content = content.replace(
    '<div style="overflow-x: auto;">\n                    <table class="data-table"',
    '<div class="table-wrapper" style="overflow-x: auto;">\n                    <table class="data-table"'
)

# 2. Add sortable headers
# Tradable
content = content.replace(
    '''<table class="data-table" id="tradable-positions-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Открыта</th>
                                <th>Рынок</th>
                                <th>Исход</th>
                                <th>Вложено</th>
                                <th>Shares</th>
                                <th>Вход</th>
                                <th>PnL</th>
                                <th>Статус</th>
                                <th>Действие</th>
                            </tr>''',
    '''<table class="data-table" id="tradable-positions-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-col="id">ID</th>
                                <th class="sortable" data-col="created_at">Открыта</th>
                                <th>Рынок</th>
                                <th class="sortable" data-col="outcome_bought">Исход</th>
                                <th class="sortable" data-col="entry_cost_usdc">Вложено</th>
                                <th class="sortable" data-col="remaining_shares">Shares</th>
                                <th class="sortable" data-col="executed_price">Вход</th>
                                <th class="sortable" data-col="realized_pnl_usdc">PnL</th>
                                <th class="sortable" data-col="position_status">Статус</th>
                                <th>Действие</th>
                            </tr>'''
)

# Resolved
content = content.replace(
    '''<table class="data-table" id="resolved-markets-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Открыта</th>
                                <th>Рынок</th>
                                <th>Исход</th>
                                <th>Вложено</th>
                                <th>Shares</th>
                                <th>PnL</th>
                                <th>Статус Позиции</th>
                                <th>Статус Погашения</th>
                                <th>Действие</th>
                            </tr>''',
    '''<table class="data-table" id="resolved-markets-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-col="id">ID</th>
                                <th class="sortable" data-col="created_at">Открыта</th>
                                <th>Рынок</th>
                                <th class="sortable" data-col="outcome_bought">Исход</th>
                                <th class="sortable" data-col="entry_cost_usdc">Вложено</th>
                                <th class="sortable" data-col="remaining_shares">Shares</th>
                                <th class="sortable" data-col="realized_pnl_usdc">PnL</th>
                                <th class="sortable" data-col="position_status">Статус Позиции</th>
                                <th class="sortable" data-col="redemption_status">Статус Погашения</th>
                                <th>Действие</th>
                            </tr>'''
)

# Archive
content = content.replace(
    '''<table class="data-table" id="archive-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Открыта</th>
                                <th>Рынок</th>
                                <th>Исход</th>
                                <th>Вложено</th>
                                <th>Shares</th>
                                <th>PnL</th>
                                <th>Статус</th>
                            </tr>''',
    '''<table class="data-table" id="archive-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-col="id">ID</th>
                                <th class="sortable" data-col="created_at">Открыта</th>
                                <th>Рынок</th>
                                <th class="sortable" data-col="outcome_bought">Исход</th>
                                <th class="sortable" data-col="entry_cost_usdc">Вложено</th>
                                <th class="sortable" data-col="remaining_shares">Shares</th>
                                <th class="sortable" data-col="realized_pnl_usdc">PnL</th>
                                <th class="sortable" data-col="position_status">Статус</th>
                            </tr>'''
)

# Requests
content = content.replace(
    '''<table class="data-table" id="requests-table">
                        <thead>
                            <tr>
                                <th>Время</th>
                                <th>Intent</th>
                                <th>Рынок</th>
                                <th>Исход</th>
                                <th>Сумма</th>
                                <th>Статус</th>
                                <th>Причина Ошибки</th>
                                <th>Действия</th>
                            </tr>''',
    '''<table class="data-table" id="requests-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-col="created_at">Время</th>
                                <th class="sortable" data-col="intent">Intent</th>
                                <th>Рынок</th>
                                <th>Исход</th>
                                <th class="sortable" data-col="target_amount_usdc">Сумма</th>
                                <th class="sortable" data-col="state">Статус</th>
                                <th>Причина Ошибки</th>
                                <th>Действия</th>
                            </tr>'''
)

# 3. Add TableManager and togglePanel JS
js_addition = """
function togglePanel(id, btn) {
    const body = document.querySelector(`#${id} .table-wrapper`);
    const pagination = document.querySelector(`#${id} .table-pagination`);
    const hidden = body.style.display === 'none';
    body.style.display = hidden ? '' : 'none';
    if (pagination) pagination.style.display = hidden ? '' : 'none';
    btn.textContent = hidden ? 'Свернуть' : 'Развернуть';
}

class TableManager {
    constructor({ tableId, pageSize = 10, columns }) {
        this.tableId = tableId;
        this.pageSize = pageSize;
        this.columns = columns; 
        this.data = [];
        this.filtered = [];
        this.page = 1;
        this.sortCol = null;
        this.sortDir = 'desc'; 
        this._buildPagination();
        this._bindHeaderClicks();
    }
    setData(rows) {
        this.data = rows;
        this.page = 1;
        this._applySort();
        this.render();
    }
    _applySort() {
        this.filtered = [...this.data];
        if (!this.sortCol) return;
        const dir = this.sortDir === 'asc' ? 1 : -1;
        this.filtered.sort((a, b) => {
            const va = a[this.sortCol] ?? '';
            const vb = b[this.sortCol] ?? '';
            if (typeof va === 'number' && typeof vb === 'number') return dir * (va - vb);
            return dir * String(va).localeCompare(String(vb));
        });
    }
    _bindHeaderClicks() {
        const table = document.getElementById(this.tableId);
        if (!table) return;
        table.querySelectorAll('th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const col = th.dataset.col;
                if (this.sortCol === col) {
                    this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
                } else {
                    this.sortCol = col;
                    this.sortDir = 'desc';
                }
                table.querySelectorAll('th.sortable').forEach(h => {
                    h.classList.remove('sort-asc', 'sort-desc');
                });
                th.classList.add('sort-' + this.sortDir);
                this._applySort();
                this.render();
            });
        });
    }
    _buildPagination() {
        const table = document.getElementById(this.tableId);
        if (!table) return;
        const wrapper = table.closest('.table-wrapper') || table.parentElement;
        const div = document.createElement('div');
        div.className = 'table-pagination';
        div.id = `${this.tableId}-pagination`;
        div.innerHTML = `
            <button id="${this.tableId}-prev" onclick="">← Назад</button>
            <span class="page-info" id="${this.tableId}-page-info">Стр. 1 из 1 (0 записей)</span>
            <button id="${this.tableId}-next" onclick="">Вперёд →</button>
            <select id="${this.tableId}-size" title="Записей на странице">
                <option value="10">10</option>
                <option value="25">25</option>
                <option value="50">50</option>
                <option value="100">100</option>
            </select>
        `;
        wrapper.appendChild(div);
        
        document.getElementById(`${this.tableId}-prev`).onclick = () => {
            if (this.page > 1) { this.page--; this.render(); }
        };
        document.getElementById(`${this.tableId}-next`).onclick = () => {
            const maxPage = Math.ceil(this.filtered.length / this.pageSize);
            if (this.page < maxPage) { this.page++; this.render(); }
        };
        document.getElementById(`${this.tableId}-size`).onchange = (e) => {
            this.pageSize = parseInt(e.target.value);
            this.page = 1;
            this.render();
        };
    }
    _updatePagination() {
        const total = this.filtered.length;
        const maxPage = Math.max(1, Math.ceil(total / this.pageSize));
        document.getElementById(`${this.tableId}-page-info`).textContent = `Стр. ${this.page} из ${maxPage} (${total} записей)`;
        document.getElementById(`${this.tableId}-prev`).disabled = this.page <= 1;
        document.getElementById(`${this.tableId}-next`).disabled = this.page >= maxPage;
    }
    getPageData() {
        const start = (this.page - 1) * this.pageSize;
        return this.filtered.slice(start, start + this.pageSize);
    }
    render() {} // Override
}

const tradableManager = new TableManager({ tableId: 'tradable-positions-table', pageSize: 10 });
tradableManager.render = function() {
    const tbody = document.querySelector('#tradable-positions-table tbody');
    const rows = this.getPageData();
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center;">Нет торгуемых позиций</td></tr>';
    } else {
        tbody.innerHTML = rows.map(p => {
            const act = p.available_actions || {};
            let actionHtml = '--';
            if (act.close) {
                actionHtml = `<button class="btn-action btn-danger" onclick="closePosition(${p.id})">Закрыть</button>`;
            } else if (act.reconcile_resolution) {
                actionHtml = `<button class="btn-action" style="background:#3182ce;" onclick="reconcileResolution(${p.id}, this)">Сверить результат</button>`;
            }
            return `
            <tr>
                <td>${escapeHtml(p.id)}</td>
                <td>${p.created_at ? escapeHtml(p.created_at.substring(11, 19)) : '--'}</td>
                <td>${escapeHtml(p.asset)} (${escapeHtml(p.market_id.substring(0, 8))}...)</td>
                <td>${escapeHtml(p.outcome_bought)}</td>
                <td>$${(p.entry_cost_usdc || p.amount_usdc || 0).toFixed(2)}</td>
                <td>${(p.remaining_shares || 0).toFixed(2)}</td>
                <td>$${(p.executed_price || 0).toFixed(3)}</td>
                <td style="color: ${p.realized_pnl_usdc >= 0 ? '#48bb78' : '#e53e3e'};">${p.realized_pnl_usdc != null ? p.realized_pnl_usdc.toFixed(2) : '--'}</td>
                <td><span class="status-badge status-live">${p.position_status}</span></td>
                <td>${actionHtml}</td>
            </tr>`;
        }).join('');
    }
    this._updatePagination();
};

const resolvedManager = new TableManager({ tableId: 'resolved-markets-table', pageSize: 10 });
resolvedManager.render = function() {
    const tbody = document.querySelector('#resolved-markets-table tbody');
    const rows = this.getPageData();
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center;">Нет завершённых рынков</td></tr>';
    } else {
        tbody.innerHTML = rows.map(p => {
            const act = p.available_actions || {};
            let actions = [];
            if (act.reconcile_resolution) {
                actions.push(`<button class="btn-action" style="background:#3182ce; font-size:0.8rem; margin:2px;" onclick="reconcileResolution(${p.id}, this)">Сверить результат</button>`);
            }
            if (act.redeem) {
                actions.push(`<button class="btn-action" style="background:#38a169; font-size:0.8rem; margin:2px;" onclick="redeemPosition(${p.id}, this)">Погасить</button>`);
            }
            if (act.reconcile_redemption) {
                actions.push(`<button class="btn-action" style="background:#3182ce; font-size:0.8rem; margin:2px;" onclick="reconcileRedemption(${p.id}, this)">Проверить погашение</button>`);
            }
            return `
            <tr>
                <td>${escapeHtml(p.id)}</td>
                <td>${p.created_at ? escapeHtml(p.created_at.substring(11, 19)) : '--'}</td>
                <td>${escapeHtml(p.asset)} (${escapeHtml(p.market_id.substring(0, 8))}...)</td>
                <td>${escapeHtml(p.outcome_bought)}</td>
                <td>$${(p.entry_cost_usdc || p.amount_usdc || 0).toFixed(2)}</td>
                <td>${(p.remaining_shares || 0).toFixed(2)}</td>
                <td style="color: ${p.realized_pnl_usdc >= 0 ? '#48bb78' : '#e53e3e'};">${p.realized_pnl_usdc != null ? p.realized_pnl_usdc.toFixed(2) : '--'}</td>
                <td><span class="status-badge status-shadow">${p.position_status}</span></td>
                <td><span class="status-badge status-shadow">${p.redemption_status || 'UNKNOWN'}</span></td>
                <td>${actions.join('')}</td>
            </tr>`;
        }).join('');
    }
    this._updatePagination();
};

const archiveManager = new TableManager({ tableId: 'archive-table', pageSize: 10 });
archiveManager.render = function() {
    const tbody = document.querySelector('#archive-table tbody');
    const rows = this.getPageData();
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">Архив пуст</td></tr>';
    } else {
        tbody.innerHTML = rows.map(p => `
            <tr>
                <td>${escapeHtml(p.id)}</td>
                <td>${p.created_at ? escapeHtml(p.created_at.substring(11, 19)) : '--'}</td>
                <td>${escapeHtml(p.asset)} (${escapeHtml(p.market_id.substring(0, 8))}...)</td>
                <td>${escapeHtml(p.outcome_bought)}</td>
                <td>$${(p.entry_cost_usdc || p.amount_usdc || 0).toFixed(2)}</td>
                <td>${(p.remaining_shares || 0).toFixed(2)}</td>
                <td style="color: ${p.realized_pnl_usdc >= 0 ? '#48bb78' : '#e53e3e'};">${p.realized_pnl_usdc != null ? p.realized_pnl_usdc.toFixed(2) : '--'}</td>
                <td><span class="status-badge status-shadow">${p.position_status}</span></td>
            </tr>`).join('');
    }
    this._updatePagination();
};

const requestsManager = new TableManager({ tableId: 'requests-table', pageSize: 10 });
requestsManager.render = function() {
    const tbody = document.querySelector('#requests-table tbody');
    const rows = this.getPageData();
    const batchBtn = document.getElementById('btn-batch-no-fill');
    
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">Нет LIVE-заявок</td></tr>';
        if(batchBtn) batchBtn.style.display = 'none';
        return;
    }
    
    let hasNoFillOptions = false;
    tbody.innerHTML = rows.map(r => {
        if (r.can_mark_no_fill) hasNoFillOptions = true;
        return `
        <tr>
            <td>${r.created_at ? escapeHtml(r.created_at.substring(11, 19)) : '--'}</td>
            <td><strong>${escapeHtml(r.intent)}</strong></td>
            <td>${escapeHtml(r.asset)}</td>
            <td>${escapeHtml(r.outcome_to_buy || r.outcome_bought || '--')}</td>
            <td>$${(r.target_amount_usdc || 0).toFixed(2)}</td>
            <td><span class="status-badge ${r.state === 'FILLED' ? 'status-ready' : 'status-shadow'}">${escapeHtml(r.state)}</span></td>
            <td style="color: #e53e3e; max-width: 200px; word-wrap: break-word;">${r.error_details && r.error_details.error_message_ru ? escapeHtml(r.error_details.error_message_ru) : (r.error_reason ? escapeHtml(r.error_reason) : '--')}</td>
            <td>
                ${r.can_mark_no_fill ? `
                    <button id="btn-no-fill-${r.id}" class="btn-action btn-danger" onclick="resolveNoFill('${r.id}')" style="padding: 4px 8px; font-size: 0.8rem;">No Fill</button>
                    <input type="checkbox" class="batch-no-fill-cb" value="${r.id}" style="margin-left: 5px;">
                ` : (r.state === 'MANUAL_REVIEW_REQUIRED' ? `<span style="font-size: 0.85rem; color: #dd6b20;">${r.review_blockers && r.review_blockers.length ? escapeHtml(r.review_blockers.join(', ')) : 'Требуется сверка'}</span>` : '')}
                ${r.available_actions && r.available_actions.includes('RECONCILE_WITH_POLYMARKET') ? `
                    <button class="btn-action" style="background: #3182ce; padding: 4px 8px; font-size: 0.8rem; margin-top: 4px;" onclick="reconcileRequest('${r.id}')">Сверить</button>
                ` : ''}
            </td>
        </tr>`;
    }).join('');
    
    if(batchBtn) batchBtn.style.display = hasNoFillOptions ? 'inline-block' : 'none';
    this._updatePagination();
};
"""

content = content.replace(
    '        const renderState = {',
    js_addition + '\n        const renderState = {'
)

# 4. Use setData in renderPositions and renderRequests
import re

# Replace the inner block of if (tradableKey !== renderState.tradable)
content = re.sub(
    r'(if \(tradableKey !== renderState\.tradable\) \{[\s\S]*?renderState\.tradable = tradableKey;)\s*const tbodyTradable[\s\S]*?\}\s*\}',
    r'\1\n                tradableManager.setData(tradable);\n            }',
    content
)

# Replace the inner block of if (resolvedKey !== renderState.resolved)
content = re.sub(
    r'(if \(resolvedKey !== renderState\.resolved\) \{[\s\S]*?renderState\.resolved = resolvedKey;)\s*const tbodyResolved[\s\S]*?\}\s*\}',
    r'\1\n                resolvedManager.setData(resolved);\n            }',
    content
)

# Replace the inner block of if (archiveKey !== renderState.archive)
content = re.sub(
    r'(if \(archiveKey !== renderState\.archive\) \{[\s\S]*?renderState\.archive = archiveKey;)\s*const tbodyArchive[\s\S]*?\}\s*\}',
    r'\1\n                archiveManager.setData(archive);\n            }',
    content
)

# Replace the inner block of if (requestsKey !== renderState.requests)
content = re.sub(
    r'(if \(requestsKey !== renderState\.requests\) \{[\s\S]*?renderState\.requests = requestsKey;)\s*const tbody[\s\S]*?batchBtn\.style\.display = [^;]*;\s*\}',
    r'\1\n                requestsManager.setData(requests);\n            }',
    content
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
