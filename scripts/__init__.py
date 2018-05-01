import logging
import sys

stderr_record_levels = [logging.CRITICAL, logging.ERROR, logging.DEBUG]
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(lambda r: r.levelno not in stderr_record_levels)
stdout_handler.setFormatter(formatter)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.INFO)
stderr_handler.addFilter(lambda r: r.levelno in stderr_record_levels)
stderr_handler.setFormatter(formatter)

logging.getLogger('qanet').setLevel(logging.INFO)
logging.getLogger('qanet').addHandler(stdout_handler)
logging.getLogger('qanet').addHandler(stderr_handler)
logging.getLogger('__main__').setLevel(logging.INFO)
logging.getLogger('__main__').addHandler(stdout_handler)
logging.getLogger('__main__').addHandler(stderr_handler)
