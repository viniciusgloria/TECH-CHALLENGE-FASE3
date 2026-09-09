"""Risco de não atingir metas futuras (2025-2030) a partir da taxa 2024 e da tendência 2023-2024."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
IMAGES = ROOT / "images" / "metas_futuras"
REPORTS = ROOT / "reports"
ANOS_FUTUROS = list(range(2025, 2031))


def ibge7(s: pd.Series) -> pd.Series:
    return s.astype("Int64").astype(str).str.zfill(7)


def main() -> None:
    integrada = pd.read_parquet(SILVER / "municipio_ano_integrada.parquet")
    evolucao = pd.read_parquet(GOLD / "evolucao_temporal.parquet")
    meta = pd.read_csv(RAW / "metamunicipal.csv")

    base = integrada.loc[integrada["ano"] == 2024].dropna(subset=["atingiu_meta"]).copy()
    base["id_municipio"] = ibge7(base["id_municipio"])
    lag = evolucao.loc[evolucao["ano"] == 2024, ["id_municipio", "taxa_ano_anterior", "variacao"]]
    lag["id_municipio"] = ibge7(lag["id_municipio"])
    base = base.merge(lag, on="id_municipio", how="left")

    meta = meta.copy()
    meta["id_municipio"] = ibge7(meta["id_municipio"])
    meta = meta.sort_values(["id_municipio", "ano"]).drop_duplicates("id_municipio", keep="last")

    colunas_meta = {ano: f"meta_alfabetizacao_{ano}" for ano in ANOS_FUTUROS}
    metas = meta[["id_municipio", *colunas_meta.values()]].copy()
    for c in colunas_meta.values():
        metas[c] = pd.to_numeric(metas[c], errors="coerce")
    base = base.merge(metas, on="id_municipio", how="left")

    # Cenário A: persistência da taxa de 2024.
    # Cenário B: tendência linear (mesmo incremento de 2023 para 2024, por ano).
    base["delta_anual"] = base["taxa_alfabetizacao"] - base["taxa_ano_anterior"]

    resumo = []
    for i, ano in enumerate(ANOS_FUTUROS, start=1):
        col = colunas_meta[ano]
        persiste = (base["taxa_alfabetizacao"] < base[col]).astype("float")
        projetada = (base["taxa_alfabetizacao"] + i * base["delta_anual"]).clip(0, 100)
        tendencia = (projetada < base[col]).astype("float")
        resumo.append(
            {
                "ano_meta": ano,
                "n": int(base[col].notna().sum()),
                "prop_risco_persistencia": round(persiste.mean(), 3),
                "prop_risco_tendencia": round(tendencia.mean(), 3),
                "meta_media": round(base[col].mean(), 2),
            }
        )
        base[f"risco_persistencia_{ano}"] = persiste
        base[f"risco_tendencia_{ano}"] = tendencia

    tabela = pd.DataFrame(resumo)
    print(tabela.to_string(index=False))

    ranking = base[
        [
            "id_municipio",
            "sigla_uf",
            "regiao",
            "taxa_alfabetizacao",
            "taxa_ano_anterior",
            "meta_ano_corrente",
            "atingiu_meta",
            "risco_persistencia_2025",
            "risco_tendencia_2025",
            "risco_persistencia_2030",
            "risco_tendencia_2030",
        ]
    ].copy()
    ranking["nao_atingiu_2024"] = (ranking["atingiu_meta"] == 0).astype(int)
    ranking = ranking.sort_values(
        ["risco_persistencia_2025", "nao_atingiu_2024", "taxa_alfabetizacao"],
        ascending=[False, False, True],
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(REPORTS / "risco_metas_futuras_resumo.csv", index=False)
    ranking.to_csv(REPORTS / "risco_metas_futuras_municipios.csv", index=False)

    por_regiao = (
        base.groupby("regiao")[["risco_persistencia_2025", "risco_persistencia_2030"]].mean().round(3)
    )
    print("\nrisco persistência por região:\n", por_regiao.to_string())
    por_regiao.to_csv(REPORTS / "risco_metas_futuras_por_regiao.csv")

    fig, ax = plt.subplots()
    ax.plot(tabela["ano_meta"], tabela["prop_risco_persistencia"], marker="o", label="persistência da taxa 2024")
    ax.plot(tabela["ano_meta"], tabela["prop_risco_tendencia"], marker="o", label="tendência 2023-2024")
    ax.set_title("Proporção de municípios abaixo da meta futura")
    ax.set_xlabel("Ano da meta")
    ax.set_ylabel("Proporção em risco")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "01_prop_risco_por_ano.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots()
    por_regiao["risco_persistencia_2025"].sort_values().plot(kind="bar", ax=ax)
    ax.set_title("Risco de não atingir a meta 2025 (persistência da taxa 2024)")
    ax.set_ylabel("Proporção")
    fig.tight_layout()
    fig.savefig(IMAGES / "02_risco_2025_por_regiao.png", dpi=140)
    plt.close()

    print("saída: reports/risco_metas_futuras_*.csv e images/metas_futuras/")


if __name__ == "__main__":
    main()
