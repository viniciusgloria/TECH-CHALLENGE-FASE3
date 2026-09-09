"""Monta a base de modelagem 2024 com Gold + enriquecimento, sem features vazadas."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"

NUM_FEATURES = [
    "taxa_ano_anterior",
    "log_populacao",
    "pib_per_capita",
    "escolas_por_mil",
    "prop_agua_potavel",
    "prop_internet",
    "prop_banda_larga",
    "prop_biblioteca",
    "prop_lab_informatica",
    "idhm_e",
    "idhm_l",
    "idhm_r",
    "meta_uf_ano_corrente",
]
CAT_FEATURES = ["sigla_uf", "regiao"]
TARGET = "nao_atingiu_meta"


def ibge7(s: pd.Series) -> pd.Series:
    return s.astype("Int64").astype(str).str.zfill(7)


def main() -> None:
    integrada = pd.read_parquet(SILVER / "municipio_ano_integrada.parquet")
    evolucao = pd.read_parquet(GOLD / "evolucao_temporal.parquet")
    pib = pd.read_csv(RAW / "ibge_pib.csv")
    pop = pd.read_csv(RAW / "ibge_populacao.csv")
    censo = pd.read_csv(RAW / "censo_escola_mun.csv")
    atlas = pd.read_csv(RAW / "atlas_idhm.csv")
    meta_uf_path = RAW / "meta_uf.csv"
    meta_br_path = RAW / "meta_brasil.csv"
    if not meta_uf_path.exists() or not meta_br_path.exists():
        raise FileNotFoundError(
            "Coloque meta_uf.csv e meta_brasil.csv em data/raw/. "
            "Consultas 5 e 6 em src/ingestao/consultas_enriquecimento.sql"
        )
    meta_uf = pd.read_csv(meta_uf_path)
    meta_br = pd.read_csv(meta_br_path)

    base = integrada.loc[integrada["ano"] == 2024].dropna(subset=["atingiu_meta"]).copy()
    base["id_municipio"] = ibge7(base["id_municipio"])
    lag = evolucao.loc[evolucao["ano"] == 2024, ["id_municipio", "taxa_ano_anterior"]]
    lag["id_municipio"] = ibge7(lag["id_municipio"])
    base = base.merge(lag, on="id_municipio", how="left")

    pib["id_municipio"] = ibge7(pib["id_municipio"])
    pop["id_municipio"] = ibge7(pop["id_municipio"])
    censo["id_municipio"] = ibge7(censo["id_municipio"])
    atlas["id_municipio"] = ibge7(atlas["id_municipio"])

    base = base.merge(pib[["id_municipio", "pib"]], on="id_municipio", how="left")
    base = base.merge(pop[["id_municipio", "populacao"]], on="id_municipio", how="left")
    base = base.merge(censo, on="id_municipio", how="left")
    base = base.merge(atlas[["id_municipio", "idhm", "idhm_e", "idhm_l", "idhm_r"]], on="id_municipio", how="left")

    meta_uf = meta_uf.loc[meta_uf["ano"] == 2024].drop_duplicates("sigla_uf", keep="last")
    meta_uf["meta_uf_ano_corrente"] = pd.to_numeric(meta_uf["meta_alfabetizacao_2024"], errors="coerce")
    base = base.merge(meta_uf[["sigla_uf", "meta_uf_ano_corrente"]], on="sigla_uf", how="left")

    meta_br = meta_br.loc[meta_br["ano"] == 2024]
    if meta_br.empty:
        meta_br = pd.read_csv(meta_br_path).sort_values("ano").tail(1)
    base["meta_brasil_ano_corrente"] = pd.to_numeric(meta_br["meta_alfabetizacao_2024"].iloc[0], errors="coerce")

    base["nao_atingiu_meta"] = (base["atingiu_meta"] == 0).astype(int)
    base["pib_per_capita"] = base["pib"] / base["populacao"]
    base["log_populacao"] = np.log1p(base["populacao"])
    base["escolas_por_mil"] = base["qtd_escolas"] / base["populacao"] * 1000

    print("linhas:", len(base))
    print("cobertura das fontes (% não nulo):")
    for c in ["pib", "populacao", "qtd_escolas", "idhm", "taxa_ano_anterior", "meta_uf_ano_corrente", "meta_brasil_ano_corrente"]:
        print(f"  {c}: {base[c].notna().mean()*100:.1f}")

    print("\ncorrelação Pearson com o alvo:")
    cols = NUM_FEATURES + [TARGET]
    print(base[cols].corr()[TARGET].sort_values().round(3).to_string())

    SILVER.mkdir(parents=True, exist_ok=True)
    dest = SILVER / "base_modelo.parquet"
    base.to_parquet(dest, index=False)
    print(f"\nsalvo: {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
