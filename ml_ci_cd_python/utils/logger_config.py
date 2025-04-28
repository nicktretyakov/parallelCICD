# Configures logging for the application.

import logging

def setup_logging():
    # TODO: Configure logging level, format, and handlers
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    pass
