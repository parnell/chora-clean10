from typing import Any

from flask import Request


def get_parameters(request: Request) -> dict[str, Any]:
    """Retrieve parameters from the request as a dictionary."""
    if request.method == 'GET':
        # Handle GET request
        return request.args.to_dict()
    elif request.method == 'POST':
        # Handle POST request
        return request.get_json() or {}
    return {}