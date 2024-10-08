from logging import getLogger
from typing import Optional, cast

from flask import Blueprint, current_app, jsonify, request
import json
from chora.app import ChoraFlask
from chora.recommender import RecommenderService

from .utils import get_parameters

log = getLogger(__name__)
bp = Blueprint("user", __name__, url_prefix="/user")


@bp.route("/recommend", methods=["GET", "POST"])
def recommend():
    app = cast(ChoraFlask, current_app)

    p = get_parameters(request)
    # Retrieve parameters from query or JSON body
    user_id = p.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    n = int(p.get("n", 10))

    # Call the recommend function with the parameters
    recommendations = app.service.recommend(
        user_id=user_id,
        n=n,
    )
    rd = json.loads(recommendations.to_json())
    log.info(f"Recommendations for user {user_id}: {rd}")

    return (
        jsonify(
            {
                "user_id": user_id,
                "recommendations": rd,
            }
        ),
        200,
    )
