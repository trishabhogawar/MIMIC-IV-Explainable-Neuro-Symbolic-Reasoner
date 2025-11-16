# load_terminologies.py
import os, sqlite3, pandas as pd

BASE = os.path.join("data", "terminologies")
RXN_RRF = os.path.join(BASE, "rxnorm", "rrf")
UMLS_DIR = os.path.join(BASE, "umls")
CCDA_XLSX = os.path.join(BASE, "ccda", "c_cda_release_20240809.xlsx")
DB_PATH = os.path.join(BASE, "terminology.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def load_rrf(path, filename, cols):
    fp = os.path.join(path, filename)
    df = pd.read_csv(fp, sep="|", header=None, dtype=str, low_memory=False)
    if df.shape[1] == len(cols) + 1:  # trailing '|'
        df = df.iloc[:, :-1]
    df.columns = cols
    return df

def main():
    con = sqlite3.connect(DB_PATH)

    # -------- RxNorm --------
    rxnconso_cols = ["RXCUI","LAT","TS","LUI","STT","SUI","ISPREF","RXAUI","SAUI","SCUI","SDUI",
                     "SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
    rxnrel_cols   = ["RXCUI1","RXAUI1","STYPE1","REL","RXCUI2","RXAUI2","STYPE2","RELA",
                     "RUI","SRUI","SAB","SL","DIR","RG","SUPPRESS","CVF"]
    rxnsat_cols   = ["RXCUI","LUI","SUI","RXAUI","STYPE","CODE","ATUI","SATUI","ATN","SAB",
                     "ATV","SUPPRESS","CVF"]

    print("RxNorm: RXNCONSO...")
    load_rrf(RXN_RRF,"RXNCONSO.RRF",rxnconso_cols).to_sql("rxnconso",con,if_exists="replace",index=False)
    print("RxNorm: RXNREL...")
    load_rrf(RXN_RRF,"RXNREL.RRF",rxnrel_cols).to_sql("rxnrel",con,if_exists="replace",index=False)
    print("RxNorm: RXNSAT...")
    load_rrf(RXN_RRF,"RXNSAT.RRF",rxnsat_cols).to_sql("rxnsat",con,if_exists="replace",index=False)

    mrconso_fp = os.path.join(UMLS_DIR,"MRCONSO.RRF")
    if os.path.exists(mrconso_fp):
        mrconso_cols = ["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI",
                        "SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
        print("UMLS: MRCONSO...")
        load_rrf(UMLS_DIR,"MRCONSO.RRF",mrconso_cols).to_sql("mrconso",con,if_exists="replace",index=False)

    if os.path.exists(CCDA_XLSX):
        print("C-CDA: Excel sheets...")
        xl = pd.ExcelFile(CCDA_XLSX)
        for sheet in xl.sheet_names:
            tname = "ccda_" + sheet.strip().lower().replace(" ","_").replace("-","_")
            xl.parse(sheet, dtype=str).to_sql(tname, con, if_exists="replace", index=False)
            print("  ->", tname)

    con.close()
    print(f"Done. SQLite at {DB_PATH}")

if __name__ == "__main__":
    main()
