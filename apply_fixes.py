import re
import os

path = "C:/Users/orlov/.gemini/antigravity/scratch/flipoly/polyflip/templates/execution.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: togglePanel null-guard
content = content.replace(
    '''function togglePanel(id, btn) {
    const body = document.querySelector(`#${id} .table-wrapper`);
    const pagination = document.querySelector(`#${id} .table-pagination`);''',
    '''function togglePanel(id, btn) {
    const body = document.querySelector(`#${id} .table-wrapper`);
    if (!body) return;
    const pagination = document.querySelector(`#${id} .table-pagination`);'''
)

# Fix 4: colCount in TableManager constructor and pagination
content = content.replace(
    '''class TableManager {
    constructor({ tableId, pageSize = 10, columns }) {
        this.tableId = tableId;
        this.pageSize = pageSize;
        this.columns = columns; ''',
    '''class TableManager {
    constructor({ tableId, pageSize = 10, columns, colCount }) {
        this.tableId = tableId;
        this.pageSize = pageSize;
        this.columns = columns; 
        this.colCount = colCount || 8;'''
)

# Apply colCount to empty states
content = content.replace(
    '''tbody.innerHTML = '<tr><td colspan="10" style="text-align: center;">Нет торгуемых позиций</td></tr>';''',
    '''tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Нет торгуемых позиций</td></tr>`;'''
)
content = content.replace(
    '''tbody.innerHTML = '<tr><td colspan="10" style="text-align: center;">Нет завершённых рынков</td></tr>';''',
    '''tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Нет завершённых рынков</td></tr>`;'''
)
content = content.replace(
    '''tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">Архив пуст</td></tr>';''',
    '''tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Архив пуст</td></tr>`;'''
)
content = content.replace(
    '''tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">Нет LIVE-заявок</td></tr>';''',
    '''tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Нет LIVE-заявок</td></tr>`;'''
)

# Fix 3: Wrap TableManager initialization in DOMContentLoaded
# Also expose them to window so renderPositions/renderRequests can access them.

# First, replace the manager instantiations
old_init_code = """
const tradableManager = new TableManager({ tableId: 'tradable-positions-table', pageSize: 10 });
tradableManager.render = function() {
    const tbody = document.querySelector('#tradable-positions-table tbody');
    const rows = this.getPageData();
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Нет торгуемых позиций</td></tr>`;
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
        tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Нет завершённых рынков</td></tr>`;
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
        tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Архив пуст</td></tr>`;
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
        tbody.innerHTML = `<tr><td colspan="${this.colCount}" style="text-align: center;">Нет LIVE-заявок</td></tr>`;
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
};"""

new_init_code = old_init_code.replace("const tradableManager", "window.tradableManager")
new_init_code = new_init_code.replace("new TableManager({ tableId: 'tradable-positions-table', pageSize: 10 });", "new TableManager({ tableId: 'tradable-positions-table', pageSize: 10, colCount: 10 });")

new_init_code = new_init_code.replace("const resolvedManager", "window.resolvedManager")
new_init_code = new_init_code.replace("new TableManager({ tableId: 'resolved-markets-table', pageSize: 10 });", "new TableManager({ tableId: 'resolved-markets-table', pageSize: 10, colCount: 10 });")

new_init_code = new_init_code.replace("const archiveManager", "window.archiveManager")
new_init_code = new_init_code.replace("new TableManager({ tableId: 'archive-table', pageSize: 10 });", "new TableManager({ tableId: 'archive-table', pageSize: 10, colCount: 8 });")

new_init_code = new_init_code.replace("const requestsManager", "window.requestsManager")
new_init_code = new_init_code.replace("new TableManager({ tableId: 'requests-table', pageSize: 10 });", "new TableManager({ tableId: 'requests-table', pageSize: 10, colCount: 8 });")

wrapped_init_code = "document.addEventListener('DOMContentLoaded', () => {\n" + "\n".join(["    " + line for line in new_init_code.split("\n")]) + "\n});"

content = content.replace(old_init_code, wrapped_init_code)

# Fix 4 for window references in renderPositions
content = content.replace("tradableManager.setData(tradable);", "if(window.tradableManager) window.tradableManager.setData(tradable);")
content = content.replace("resolvedManager.setData(resolved);", "if(window.resolvedManager) window.resolvedManager.setData(resolved);")
content = content.replace("archiveManager.setData(archive);", "if(window.archiveManager) window.archiveManager.setData(archive);")
content = content.replace("requestsManager.setData(requests);", "if(window.requestsManager) window.requestsManager.setData(requests);")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
