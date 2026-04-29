def setup_app_logger() -> logging.Logger:
    env_level = os.environ.get("CODE_SIM_LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, env_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging initialized.")
    return logger