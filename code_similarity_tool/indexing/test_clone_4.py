def initialize_system_logger():
    logger = logging.getLogger(__name__)
    level_str = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    
    if hasattr(logging, level_str):
        logger.setLevel(getattr(logging, level_str))
        
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger