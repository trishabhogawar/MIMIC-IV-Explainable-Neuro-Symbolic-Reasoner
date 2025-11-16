# terminology.py
import os, sqlite3, pandas as pd
DB_PATH = os.path.join("data","terminologies","terminology.db")

def _q(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    return df

def rxnorm_search_by_name(name_like, max_rows=50):
    sql = """
    SELECT RXCUI, SAB, TTY, STR
    FROM rxnconso
    WHERE SAB='RXNORM' AND UPPER(STR) LIKE UPPER(?)
    LIMIT ?
    """
    return _q(sql, (f"%{name_like}%", max_rows))

def rxnorm_best_rxcui_for_drugname(drug_str):
    sql = """
    SELECT RXCUI, TTY, STR
    FROM rxnconso
    WHERE SAB='RXNORM' AND UPPER(STR)=UPPER(?)
    ORDER BY CASE TTY WHEN 'SCD' THEN 1 WHEN 'SBD' THEN 2 WHEN 'IN' THEN 3 ELSE 9 END
    LIMIT 1
    """
    df = _q(sql, (drug_str,))
    return df.iloc[0].to_dict() if not df.empty else None

def rxnorm_ingredients(rxcui):
    sql = """
    SELECT DISTINCT c2.RXCUI AS ing_rxcui, c2.STR AS ing_name
    FROM rxnrel r
    JOIN rxnconso c2 ON c2.RXCUI = r.RXCUI2 AND c2.SAB='RXNORM'
    WHERE r.RXCUI1=? AND r.RELA='has_ingredient' AND r.SAB='RXNORM'
    """
    return _q(sql, (str(rxcui),))
