import inspect
import typing
from fastapi import Request
from polyflip.api.settings import update_settings_bulk, update_setting, update_security_setting

def test_settings_endpoints_accept_none_request():
    for fn in [update_settings_bulk, update_setting, update_security_setting]:
        sig = inspect.signature(fn)
        ann = sig.parameters['request'].annotation
        
        # Должен быть Optional[Request], а не Request
        origin = typing.get_origin(ann)
        args = typing.get_args(ann)
        
        assert origin is typing.Union, f"{fn.__name__}: request должен быть Optional[Request]"
        assert type(None) in args, f"{fn.__name__}: None должен быть допустимым типом для request"
        assert Request in args, f"{fn.__name__}: Request должен быть допустимым типом для request"
