"""EDA da base município-ano e insumos para a definição do alvo."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
IMAGES = ROOT / "images" / "eda"
REPORTS = ROOT / "reports"

plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def salvar_fig(nome: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    caminho = IMAGES / nome
    plt.tight_layout()
    plt.savefig(caminho, dpi=140)
    plt.close()
    print(f"figura: {caminho.relative_to(ROOT)}")


def main() -> None:
    integrada = pd.read_parquet(SILVER / "municipio_ano_integrada.parquet")
    evolucao = pd.read_parquet(GOLD / "evolucao_temporal.parquet")

    integrada["atingiu_meta"] = integrada["atingiu_meta"].astype("float")
    base_2024 = integrada.loc[integrada["ano"] == 2024].copy()
    alvo = base_2024.dropna(subset=["atingiu_meta"]).copy()

    lag = evolucao.loc[evolucao["ano"] == 2024, ["id_municipio", "taxa_ano_anterior", "variacao"]]
    alvo = alvo.merge(lag, on="id_municipio", how="left")

    print("=== qualidade ===")
    print("linhas:", len(integrada), "municípios:", integrada["id_municipio"].nunique())
    print("chave duplicada:", integrada.duplicated(["ano", "id_municipio"]).sum())
    print("nulos %:\n", (integrada.isna().mean() * 100).round(2).to_string())
    print("municípios só 2023:", integrada.groupby("id_municipio")["ano"].nunique().eq(1).sum())

    print("\n=== descritivo numérico ===")
    print(integrada[["taxa_alfabetizacao", "media_portugues", "meta_ano_corrente", "gap_meta"]].describe().round(2).to_string())

    print("\n=== correlação Pearson (2024 com alvo) ===")
    cols = ["taxa_alfabetizacao", "media_portugues", "meta_ano_corrente", "gap_meta", "taxa_ano_anterior", "variacao"]
    print(alvo[cols].corr().round(3).to_string())
    print("corr taxa x media_portugues 2023:", integrada.loc[integrada["ano"] == 2023, ["taxa_alfabetizacao", "media_portugues"]].corr().iloc[0, 1])

    print("\n=== alvo 2024 ===")
    print(alvo["atingiu_meta"].value_counts().to_string())
    print("proporção positiva:", round(alvo["atingiu_meta"].mean(), 4))

    print("\n=== taxa média por ano ===")
    print(integrada.groupby("ano")["taxa_alfabetizacao"].agg(["mean", "median", "std"]).round(2).to_string())

    print("\n=== por região (2024, com alvo) ===")
    por_regiao = (
        alvo.groupby("regiao")
        .agg(
            n=("id_municipio", "count"),
            taxa_media=("taxa_alfabetizacao", "mean"),
            meta_media=("meta_ano_corrente", "mean"),
            prop_na_meta=("atingiu_meta", "mean"),
            taxa_2023=("taxa_ano_anterior", "mean"),
        )
        .round(2)
        .sort_values("prop_na_meta")
    )
    print(por_regiao.to_string())

    print("\n=== por UF (2024) ===")
    por_uf = (
        alvo.groupby(["sigla_uf", "regiao"])
        .agg(
            n=("id_municipio", "count"),
            taxa_media=("taxa_alfabetizacao", "mean"),
            prop_na_meta=("atingiu_meta", "mean"),
        )
        .round(3)
        .sort_values("prop_na_meta")
    )
    print(por_uf.to_string())

    naive = ((alvo["taxa_ano_anterior"] >= alvo["meta_ano_corrente"]).astype(float))
    print("\n=== baseline naive (2023 >= meta 2024) ===")
    print("cobertura lag:", alvo["taxa_ano_anterior"].notna().mean())
    mask = alvo["taxa_ano_anterior"].notna()
    y = alvo.loc[mask, "atingiu_meta"]
    yhat = naive.loc[mask]
    acc = (y == yhat).mean()
    print("acurácia naive:", round(acc, 4))
    print("proporção positiva naive:", round(yhat.mean(), 4))

    sem_meta = base_2024.loc[base_2024["meta_ano_corrente"].isna()]
    print("\n=== 2024 sem meta ===")
    print("n:", len(sem_meta), "taxa média:", round(sem_meta["taxa_alfabetizacao"].mean(), 2))
    print(sem_meta["regiao"].value_counts().to_string())

    # figuras
    fig, ax = plt.subplots()
    integrada.boxplot(column="taxa_alfabetizacao", by="ano", ax=ax)
    ax.set_title("Taxa de alfabetização por ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Taxa (%)")
    plt.suptitle("")
    salvar_fig("01_taxa_por_ano.png")

    fig, ax = plt.subplots()
    alvo["taxa_alfabetizacao"].hist(bins=30, ax=ax, alpha=0.8)
    ax.axvline(alvo["taxa_alfabetizacao"].median(), color="black", linestyle="--", label="mediana")
    ax.set_title("Distribuição da taxa em 2024 (municípios com meta)")
    ax.set_xlabel("Taxa (%)")
    ax.legend()
    salvar_fig("02_hist_taxa_2024.png")

    fig, ax = plt.subplots()
    alvo.boxplot(column="taxa_alfabetizacao", by="regiao", ax=ax)
    ax.set_title("Taxa 2024 por região")
    ax.set_xlabel("Região")
    ax.set_ylabel("Taxa (%)")
    plt.suptitle("")
    salvar_fig("03_taxa_por_regiao.png")

    fig, ax = plt.subplots()
    ordem = por_regiao.index.tolist()
    por_regiao.loc[ordem, "prop_na_meta"].plot(kind="bar", ax=ax)
    ax.set_title("Proporção de municípios que atingiram a meta em 2024")
    ax.set_ylabel("Proporção")
    ax.set_xlabel("Região")
    salvar_fig("04_prop_meta_por_regiao.png")

    fig, ax = plt.subplots()
    ax.scatter(alvo["media_portugues"], alvo["taxa_alfabetizacao"], s=8, alpha=0.25)
    ax.set_title("Taxa vs média de português (2024)")
    ax.set_xlabel("Média português")
    ax.set_ylabel("Taxa (%)")
    salvar_fig("05_taxa_vs_media_portugues.png")

    fig, ax = plt.subplots()
    ax.scatter(alvo["meta_ano_corrente"], alvo["taxa_alfabetizacao"], c=alvo["atingiu_meta"], s=8, alpha=0.3, cmap="bwr")
    ax.plot([0, 100], [0, 100], color="black", linewidth=1)
    ax.set_title("Taxa vs meta 2024 (azul = não atingiu, vermelho = atingiu)")
    ax.set_xlabel("Meta (%)")
    ax.set_ylabel("Taxa (%)")
    salvar_fig("06_taxa_vs_meta.png")

    fig, ax = plt.subplots()
    ax.scatter(alvo["taxa_ano_anterior"], alvo["taxa_alfabetizacao"], s=8, alpha=0.25)
    ax.plot([0, 100], [0, 100], color="black", linewidth=1)
    ax.set_title("Taxa 2024 vs taxa 2023")
    ax.set_xlabel("Taxa 2023 (%)")
    ax.set_ylabel("Taxa 2024 (%)")
    salvar_fig("07_taxa_2024_vs_2023.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    por_uf.reset_index().sort_values("prop_na_meta").plot(
        x="sigla_uf", y="prop_na_meta", kind="bar", ax=ax, legend=False
    )
    ax.set_title("Proporção de municípios na meta por UF (2024)")
    ax.set_ylabel("Proporção")
    ax.set_xlabel("UF")
    salvar_fig("08_prop_meta_por_uf.png")

    # tabelas auxiliares para o relatório
    REPORTS.mkdir(parents=True, exist_ok=True)
    por_regiao.to_csv(REPORTS / "eda_por_regiao.csv")
    por_uf.to_csv(REPORTS / "eda_por_uf.csv")
    print("tabelas: reports/eda_por_regiao.csv e reports/eda_por_uf.csv")


if __name__ == "__main__":
    main()
