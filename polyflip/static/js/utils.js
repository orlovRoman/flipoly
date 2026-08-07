/**
 * Общие утилиты для дашборда
 */

function formatDateUTC(dateStr) {
    if (!dateStr) return '--';
    const d = new Date(dateStr);
    if (isNaN(d)) return '--';
    return d.toLocaleString("ru-RU", { 
        timeZone: "UTC", 
        month: "short", 
        day: "numeric", 
        hour: "2-digit", 
        minute: "2-digit", 
        second: "2-digit" 
    }) + ' UTC';
}
