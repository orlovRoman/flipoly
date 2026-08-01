class GatewayError(Exception):
    """Базовый класс исключений шлюзов исполнения."""


class GatewayUnavailable(GatewayError):
    """Шлюз временно недоступен."""


class GatewayOrderRejected(GatewayError):
    """Биржа или клиентский SDK детерминированно отклонили ордер."""


class GatewaySubmissionUnknown(GatewayError):
    """Результат отправки ордера неизвестен из-за разрыва связи."""
