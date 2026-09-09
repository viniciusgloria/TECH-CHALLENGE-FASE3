"""Clusterização de municípios (pergunta: regiões com padrões semelhantes)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
IMAGES = ROOT / "images" / "clusters"
REPORTS = ROOT / "reports"


def ibge7(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)


def main() -> None:
    integrada = pd.read_parquet(SILVER / "municipio_ano_integrada.parquet")
    evolucao = pd.read_parquet(GOLD / "evolucao_temporal.parquet")
    base = integrada.loc[integrada["ano"] == 2024].dropna(subset=["atingiu_meta"]).copy()
    lag = evolucao.loc[evolucao["ano"] == 2024, ["id_municipio", "taxa_ano_anterior"]]
    base = base.merge(lag, on="id_municipio", how="left")
    base["id_municipio"] = ibge7(base["id_municipio"])

    num = ["taxa_alfabetizacao", "taxa_ano_anterior", "meta_ano_corrente"]
    X = base[num].copy()
    X = X.join(pd.get_dummies(base["regiao"], prefix="reg", drop_first=False))
    X = X.dropna()
    usados = base.loc[X.index].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    linhas = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        rotulos = km.fit_predict(Xs)
        sil = silhouette_score(Xs, rotulos)
        linhas.append({"k": k, "silhouette": round(sil, 4), "inercia": round(km.inertia_, 1)})
        print(f"k={k} silhouette={sil:.4f}")

    grade = pd.DataFrame(linhas)
    k_melhor = int(grade.sort_values("silhouette", ascending=False).iloc[0]["k"])
    print("k escolhido por silhouette:", k_melhor)

    modelo = KMeans(n_clusters=k_melhor, random_state=42, n_init=10)
    usados["cluster"] = modelo.fit_predict(Xs)

    perfil = (
        usados.groupby("cluster")
        .agg(
            n=("id_municipio", "count"),
            taxa_media=("taxa_alfabetizacao", "mean"),
            taxa_2023=("taxa_ano_anterior", "mean"),
            meta_media=("meta_ano_corrente", "mean"),
            prop_nao_atingiu=("atingiu_meta", lambda s: (s == 0).mean()),
        )
        .round(2)
    )
    mix_regiao = pd.crosstab(usados["cluster"], usados["regiao"], normalize="index").round(2)
    print("\nperfil:\n", perfil.to_string())
    print("\nmix regional:\n", mix_regiao.to_string())

    IMAGES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    grade.to_csv(REPORTS / "clusters_silhouette.csv", index=False)
    perfil.to_csv(REPORTS / "clusters_perfil.csv")
    usados[["id_municipio", "sigla_uf", "regiao", "cluster", "taxa_alfabetizacao", "atingiu_meta"]].to_csv(
        REPORTS / "clusters_municipios.csv", index=False
    )

    fig, ax = plt.subplots()
    ax.plot(grade["k"], grade["silhouette"], marker="o")
    ax.set_title("Silhouette por K (K-Means)")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(IMAGES / "01_silhouette.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(usados["meta_ano_corrente"], usados["taxa_alfabetizacao"], c=usados["cluster"], s=8, alpha=0.35, cmap="tab10")
    ax.plot([0, 100], [0, 100], color="black", linewidth=1)
    ax.set_title(f"Clusters k={k_melhor}: taxa vs meta 2024")
    ax.set_xlabel("Meta (%)")
    ax.set_ylabel("Taxa (%)")
    fig.tight_layout()
    fig.savefig(IMAGES / "02_taxa_vs_meta_cluster.png", dpi=140)
    plt.close()

    print("saída: reports/clusters_*.csv e images/clusters/")


if __name__ == "__main__":
    main()
