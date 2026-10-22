import sqlite3

from app.database.seed import seed_database


def test_seed_is_idempotent_and_relational(tmp_path):
    db = tmp_path / "x.db"
    a, b = seed_database(db), seed_database(db)
    assert a == b and a["customers"] >= 30 and a["orders"] >= 100
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    assert conn.execute("PRAGMA foreign_key_list(orders)").fetchall()
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {r[0] for r in conn.execute("SELECT DISTINCT status FROM orders")} == {"completed", "pending", "cancelled", "refunded"}
    first = conn.execute("SELECT SUM(amount) FROM orders").fetchone()[0]
    seed_database(db)
    assert sqlite3.connect(db).execute("SELECT SUM(amount) FROM orders").fetchone()[0] == first
