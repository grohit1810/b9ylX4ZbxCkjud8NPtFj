"""
Script to run data ingestion pipeline.

Usage:
    python scripts/run_ingestion.py

    # With verbose logging:
    VERBOSE=true python scripts/run_ingestion.py

    # With custom log level:
    LOG_LEVEL=DEBUG python scripts/run_ingestion.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_db.pipeline import main

if __name__ == "__main__":
    main()
