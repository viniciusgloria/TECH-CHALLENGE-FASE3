"""Reconstrói a camada Gold da Fase 2 em pandas, a partir dos CSVs extraídos do BigQuery.

Lógica alinhada a:
- FASE 2/TECH CHALLENGE/src/silver/02_carga_silver.py
- FASE 2/TECH CHALLENGE/src/gold/03_carga_gold.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold"
SILVER = ROOT / "data" / "silver"

ANOS_META = list(range(2024, 2031))

UFS = [
    ("11", "RO", "Rondônia", "Norte"),
    ("12", "AC", "Acre", "Norte"),
    ("13", "AM", "Amazonas", "Norte"),
    ("14", "RR", "Roraima", "Norte"),
    ("15", "PA", "Pará", "Norte"),
    ("16", "AP", "Amapá", "Norte"),
    ("17", "TO", "Tocantins", "Norte"),
    ("21", "MA", "Maranhão", "Nordeste"),
    ("22", "PI", "Piauí", "Nordeste"),
    ("23", "CE", "Ceará", "Nordeste"),
    ("24", "RN", "Rio Grande do Norte", "Nordeste"),
    ("25", "PB", "Paraíba", "Nordeste"),
    ("26", "PE", "Pernambuco", "Nordeste"),
    ("27", "AL", "Alagoas", "Nordeste"),
    ("28", "SE", "Sergipe", "Nordeste"),
    ("29", "BA", "Bahia", "Nordeste"),
    ("31", "MG", "Minas Gerais", "Sudeste"),
    ("32", "ES", "Espírito Santo", "Sudeste"),
    ("33", "RJ", "Rio de Janeiro", "Sudeste"),
    ("35", "SP", "São Paulo", "Sudeste"),
    ("41", "PR", "Paraná", "Sul"),
    ("42", "SC", "Santa Catarina", "Sul"),
    ("43", "RS", "Rio Grande do Sul", "Sul"),
    ("50", "MS", "Mato Grosso do Sul", "Centro-Oeste"),
    ("51", "MT", "Mato Grosso", "Centro-Oeste"),
    ("52", "GO", "Goiás", "Centro-Oeste"),
    ("53", "DF", "Distrito Federal", "Centro-Oeste"),
]


def ibge7(series: pd.Series) -> pd.Series:
    return series.astype("Int64").astype(str).str.zfill(7)


def salvar(df: pd.DataFrame, pasta: Path, nome: str) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{nome}.parquet"
    df.to_parquet(caminho, index=False)
    print(f"{nome}: {len(df):,} linhas -> {caminho.relative_to(ROOT)}")


def main() -> None:
    indicador_path = RAW / "indicadormunicipal.csv"
    meta_path = RAW / "metamunicipal.csv"
    if not indicador_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"CSVs não encontrados em {RAW}. Esperados: indicadormunicipal.csv e metamunicipal.csv"
        )

    indicador = pd.read_csv(indicador_path)
    meta = pd.read_csv(meta_path)

    dim_uf = pd.DataFrame(UFS, columns=["codigo_uf", "sigla_uf", "nome_uf", "regiao"])

    fato = indicador.copy()
    fato["ano"] = fato["ano"].astype(int)
    fato["id_municipio"] = ibge7(fato["id_municipio"])
    fato["taxa_alfabetizacao"] = pd.to_numeric(fato["taxa_alfabetizacao"], errors="coerce")
    fato["media_portugues"] = pd.to_numeric(fato["media_portugues"], errors="coerce")
    fato["codigo_uf"] = fato["id_municipio"].str[:2]
    fato = (
        fato.groupby(["ano", "id_municipio", "codigo_uf"], as_index=False)
        .agg(taxa_alfabetizacao=("taxa_alfabetizacao", "mean"), media_portugues=("media_portugues", "mean"))
        .merge(dim_uf, on="codigo_uf", how="left")
    )

    meta = meta.copy()
    meta["ano"] = meta["ano"].astype(int)
    meta["id_municipio"] = ibge7(meta["id_municipio"])
    meta = meta.sort_values(["id_municipio", "ano"]).drop_duplicates("id_municipio", keep="last")

    meta_long = meta.melt(
        id_vars=["id_municipio"],
        value_vars=[f"meta_alfabetizacao_{ano}" for ano in ANOS_META],
        var_name="ano_meta_col",
        value_name="meta_taxa",
    )
    meta_long["ano_meta"] = meta_long["ano_meta_col"].str.extract(r"(\d{4})").astype(int)
    meta_long["meta_taxa"] = pd.to_numeric(meta_long["meta_taxa"], errors="coerce")
    meta_long = meta_long[["id_municipio", "ano_meta", "meta_taxa"]]

    integrada = fato.merge(
        meta_long,
        left_on=["id_municipio", "ano"],
        right_on=["id_municipio", "ano_meta"],
        how="left",
    )
    integrada = integrada.drop(columns=["ano_meta"])
    integrada = integrada.rename(columns={"meta_taxa": "meta_ano_corrente"})
    integrada["gap_meta"] = integrada["taxa_alfabetizacao"] - integrada["meta_ano_corrente"]
    integrada["atingiu_meta"] = (integrada["gap_meta"] >= 0).astype("Int64")
    integrada.loc[integrada["meta_ano_corrente"].isna(), "atingiu_meta"] = pd.NA
    integrada = integrada.drop_duplicates()

    GOLD.mkdir(parents=True, exist_ok=True)
    SILVER.mkdir(parents=True, exist_ok=True)
    salvar(dim_uf, SILVER, "dim_uf")
    salvar(fato, SILVER, "fato_municipio")
    salvar(meta_long, SILVER, "meta_municipio_long")
    salvar(integrada, SILVER, "municipio_ano_integrada")

    indicador_recente = (
        integrada.sort_values(["id_municipio", "ano"]).groupby("id_municipio", as_index=False).tail(1)
    )
    salvar(indicador_recente, GOLD, "indicador_por_municipio")

    meta_versus_resultado = integrada.loc[
        integrada["meta_ano_corrente"].notna(),
        ["ano", "id_municipio", "sigla_uf", "regiao", "taxa_alfabetizacao", "meta_ano_corrente", "gap_meta", "atingiu_meta"],
    ].copy()
    salvar(meta_versus_resultado, GOLD, "meta_versus_resultado")

    evolucao = integrada[["ano", "id_municipio", "sigla_uf", "taxa_alfabetizacao"]].sort_values(
        ["id_municipio", "ano"]
    )
    evolucao["taxa_ano_anterior"] = evolucao.groupby("id_municipio")["taxa_alfabetizacao"].shift(1)
    evolucao["variacao"] = evolucao["taxa_alfabetizacao"] - evolucao["taxa_ano_anterior"]
    salvar(evolucao, GOLD, "evolucao_temporal")

    resumo_uf = (
        integrada.groupby(["ano", "sigla_uf", "regiao"], as_index=False)
        .agg(
            qtd_municipios=("id_municipio", "count"),
            taxa_media=("taxa_alfabetizacao", "mean"),
            prop_municipios_na_meta=("atingiu_meta", "mean"),
        )
    )
    resumo_uf["taxa_media"] = resumo_uf["taxa_media"].round(2)
    resumo_uf["prop_municipios_na_meta"] = resumo_uf["prop_municipios_na_meta"].round(3)
    salvar(resumo_uf, GOLD, "resumo_por_uf")

    print("\n--- conferência ---")
    print("anos:", sorted(integrada["ano"].unique().tolist()))
    print("municípios:", integrada["id_municipio"].nunique())
    print("meta nula por ano:")
    print(integrada.groupby("ano")["meta_ano_corrente"].apply(lambda s: int(s.isna().sum())).to_string())
    alvo = integrada.dropna(subset=["atingiu_meta"])
    print("linhas com alvo binário:", len(alvo))
    print("atingiu_meta:")
    print(alvo["atingiu_meta"].value_counts(dropna=False).to_string())
    print("sigla_uf nula:", int(integrada["sigla_uf"].isna().sum()))
    print("taxa fora de [0,100]:", int(((integrada["taxa_alfabetizacao"] < 0) | (integrada["taxa_alfabetizacao"] > 100)).sum()))


if __name__ == "__main__":
    main()
