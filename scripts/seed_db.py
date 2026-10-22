"""Create the schema and load deterministic sample data (idempotent)."""
import sys

from app.core.config import get_settings
from app.database.seed import seed_database


def main() -> int:
    counts = seed_database(get_settings().sqlite_path)
    print(f"seeded {get_settings().sqlite_path}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
