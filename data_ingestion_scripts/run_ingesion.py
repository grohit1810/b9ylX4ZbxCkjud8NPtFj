import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_db.pipeline import main as sql_main
from vector_db.pipeline import main as vector_main

if __name__ == "__main__":
    sql_main()
    vector_main()