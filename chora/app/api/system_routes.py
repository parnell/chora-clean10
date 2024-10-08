import logging

from flask import Blueprint, jsonify

log = logging.getLogger(__name__)

bp = Blueprint("system", __name__, url_prefix="/system")


@bp.route("/heartbeat", methods=["GET", "POST"])
def heartbeat():
    log.debug(f"serve::heartbeat")
    return jsonify({"status": "OK"}), 200
