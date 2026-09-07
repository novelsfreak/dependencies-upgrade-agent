"""
Dumb, deliberate migration runner.

Reads db/migrations/*.sql in filename order, applies any not yet recorded
in schema_migrations, and records each one in the same transaction as the
migration itself — so a crash mid-migration never leaves a half-applied,
untracked file.

No rollback, no down-migrations. You don't need them yet.
"""
import os
import sys
import psycopg

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "db", "migrations")
DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


def ensure_migrations_table(conn: psycopg.Connection) -> None:
    conn.execute("""
        create table if not exists schema_migrations (
            filename    text primary key,
            applied_at  timestamptz not null default now()
        )
    """)


def applied_migrations(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute("select filename from schema_migrations").fetchall()
    return {r[0] for r in rows}


def pending_migrations(already_applied: set[str]) -> list[str]:
    files = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql"))
    return [f for f in files if f not in already_applied]


def apply_migration(conn: psycopg.Connection, filename: str) -> None:
    path = os.path.join(MIGRATIONS_DIR, filename)
    with open(path, "r") as f:
        sql = f.read()

    print(f"applying {filename} ...")
    with conn.transaction():
        conn.execute(sql)  # type: ignore[arg-type]
        conn.execute(
            "insert into schema_migrations (filename) values (%s)", (filename,)
        )
    print(f"  ok")


def main() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        ensure_migrations_table(conn)
        already = applied_migrations(conn)
        pending = pending_migrations(already)

        if not pending:
            print("nothing to apply")
            return

        for filename in pending:
            apply_migration(conn, filename)

        print(f"applied {len(pending)} migration(s)")


if __name__ == "__main__":
    try:
        main()
    except psycopg.OperationalError as e:
        print(f"could not connect to database: {e}", file=sys.stderr)
        print(f"DSN used: {DSN}", file=sys.stderr)
        sys.exit(1)
