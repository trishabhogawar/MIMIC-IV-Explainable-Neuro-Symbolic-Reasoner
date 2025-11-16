import os, getpass
import pandas as pd
from sqlalchemy import create_engine, text

# --- DB connection settings ---
DB_USER = os.environ.get("PGUSER") or getpass.getuser()
DB_NAME = "mimiciv"
DB_HOST = "localhost"
DB_PORT = 5432

db_url = f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url, future=True)

queries = [
    ("patients",      text("SELECT COUNT(*) AS n FROM mimiciv_hosp.patients;")),
    ("admissions",    text("SELECT COUNT(*) AS n FROM mimiciv_hosp.admissions;")),
    ("icustays",      text("SELECT COUNT(*) AS n FROM mimiciv_icu.icustays;")),
    ("creatinine_10", text("""
        SELECT subject_id, hadm_id, charttime, valuenum AS creatinine_mg_dl
        FROM mimiciv_hosp.labevents
        WHERE itemid = 50912
        ORDER BY charttime
        LIMIT 10;
    """)),
]

print(f"Connecting as user={DB_USER} to db={DB_NAME} ...")
with engine.connect() as conn:
    for label, q in queries:
        df = pd.read_sql(q, conn)
        print(f"\n== {label} ==")
        print(df.to_string(index=False))
print("\nAll queries ran successfully.")
