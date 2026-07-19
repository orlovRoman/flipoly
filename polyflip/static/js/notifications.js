// Утилиты для работы с Notification API в браузере

function requestNotificationPermission() {
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
}

function showNotification(title, body) {
  if ("Notification" in window && Notification.permission === "granted") {
    try {
      new Notification(title, {
        body: body,
        icon: "/static/img/favicon.ico"
      });
    } catch (e) {
      console.error("Failed to show notification", e);
    }
  }
}

// Экспортируем в window для глобального доступа
window.requestNotificationPermission = requestNotificationPermission;
window.showNotification = showNotification;
