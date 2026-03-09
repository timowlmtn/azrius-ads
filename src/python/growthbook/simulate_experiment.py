import os

import logging
from collections import Counter
from growthbook import GrowthBook

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)

logger = logging.getLogger("growthbook.debug")

# -------------------------------------------------------------------
# GrowthBook config
# -------------------------------------------------------------------
API_HOST = os.getenv("GROWTHBOOK_API_HOST", "http://localhost:3100")
CLIENT_KEY = os.getenv("GROWTHBOOK_CLIENT_KEY")  # must match dev/prod env

FEATURE_KEY = os.getenv("GROWTHBOOK_FEATURE_KEY", "button-color-feature")
FALLBACK = os.getenv("GROWTHBOOK_FALLBACK_VALUE", "blue")


def explain_eval(gb: GrowthBook, feature_key: str) -> str:
    """
    DEBUG-level explanation of *why* a feature value was chosen.
    Runs on every request via evalFeature().
    """
    res = gb.eval_feature(feature_key)

    logger.debug("----- FEATURE EVAL START -----")
    logger.debug("feature=%s", feature_key)

    logger.debug("value=%s", res.value)
    logger.debug("source=%s", res.source)  # default | force | experiment

    # Experiment details (only present for experiments)
    if getattr(res, "experiment", None):
        logger.debug("experiment.key=%s", res.experiment.key)

    if getattr(res, "experimentResult", None):
        er = res.experimentResult
        logger.debug("variation.key=%s", er.key)
        logger.debug("variation.value=%s", er.value)
        logger.debug("in_experiment=%s", getattr(er, "inExperiment", None))

    # Optional / SDK-version-dependent fields
    for attr in ("ruleId", "bucket", "hashAttribute", "hashValue"):
        if hasattr(res, attr):
            logger.debug("%s=%s", attr, getattr(res, attr))

    logger.debug("----- FEATURE EVAL END -----\n")
    return res.value


def run_for_user(user_id: str) -> str:
    gb = GrowthBook(
        api_host=API_HOST,
        client_key=CLIENT_KEY,
        attributes={"id": user_id},
    )

    gb.load_features()

    value = explain_eval(gb, FEATURE_KEY)

    if value is None:
        value = FALLBACK

    gb.destroy()
    return value


# -------------------------------------------------------------------
# Simulate many requests
# -------------------------------------------------------------------
counts = Counter(run_for_user(f"experiment-{str(i)}") for i in range(1, 501))
logger.info("FINAL COUNTS: %s", dict(counts))
