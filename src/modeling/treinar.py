"""Pipeline supervisionado: ColumnTransformer + modelos + SHAP."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver"
IMAGES = ROOT / "images" / "modelagem"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42

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


def metricas(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
    }


def preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, NUM_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ],
        remainder="drop",
    )


def main() -> None:
    df = pd.read_parquet(SILVER / "base_modelo.parquet")
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    print("n=", len(df), "positivos=", int(y.sum()), f"({y.mean():.3f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidatos = {
        "dummy_mais_frequente": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "regressao_logistica": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [150], "clf__max_depth": [8, 16], "clf__min_samples_leaf": [2, 5]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [3, 6], "clf__learning_rate": [0.05, 0.1]},
        ),
    }

    linhas = []
    melhor_nome = None
    melhor_f1 = -1
    melhor_pipe = None
    melhor_tree = None
    melhor_tree_f1 = -1

    for nome, (est, grade) in candidatos.items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", est)])
        if grade:
            busca = GridSearchCV(pipe, grade, cv=cv, scoring="f1", n_jobs=1, refit=True)
            busca.fit(X_train, y_train)
            modelo = busca.best_estimator_
            cv_f1 = round(busca.best_score_, 4)
            print(nome, "best_params=", busca.best_params_, "cv_f1=", cv_f1)
        else:
            pipe.fit(X_train, y_train)
            modelo = pipe
            cv_f1 = np.nan
            print(nome, "baseline sem busca")

        y_pred = modelo.predict(X_test)
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)
        m = metricas(y_test, y_pred, y_proba)
        m["modelo"] = nome
        m["cv_f1"] = cv_f1
        linhas.append(m)
        if m["f1"] > melhor_f1:
            melhor_f1 = m["f1"]
            melhor_nome = nome
            melhor_pipe = modelo
        if nome in {"random_forest", "hist_gradient_boosting"} and m["f1"] > melhor_tree_f1:
            melhor_tree = modelo
            melhor_tree_f1 = m["f1"]

    tabela = pd.DataFrame(linhas).set_index("modelo")
    print("\nmétricas no teste:\n", tabela.to_string())
    print("\nmelhor no teste:", melhor_nome)
    print(classification_report(y_test, melhor_pipe.predict(X_test), digits=3))

    IMAGES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(REPORTS / "metricas_modelos.csv")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(melhor_pipe, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Matriz de confusão no teste ({melhor_nome})")
    fig.tight_layout()
    fig.savefig(IMAGES / "01_matriz_confusao.png", dpi=140)
    plt.close()

    explainer_model = melhor_pipe
    prep_f = explainer_model.named_steps["prep"]
    clf = explainer_model.named_steps["clf"]
    X_test_t = prep_f.transform(X_test)
    nomes = prep_f.get_feature_names_out()

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, prep_f.transform(X_train))
        sv = explainer.shap_values(X_test_t)
    else:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_test_t)
    if hasattr(sv, "values"):
        sv = sv.values
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    nomes = np.array(nomes)
    if sv.shape[1] != len(nomes):
        nomes = nomes[: sv.shape[1]]

    shap.summary_plot(sv, X_test_t, feature_names=nomes, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(IMAGES / "02_shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    shap.summary_plot(sv, X_test_t, feature_names=nomes, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(IMAGES / "03_shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()

    imp = pd.Series(np.abs(sv).mean(axis=0), index=nomes).sort_values(ascending=False)
    print("\nSHAP médio absoluto (top 15):")
    print(imp.head(15).round(4).to_string())
    imp.to_csv(REPORTS / "shap_importancia.csv")

    if isinstance(clf, LogisticRegression):
        coefs = pd.Series(clf.coef_[0], index=nomes).sort_values()
        print("\ncoeficientes (negativo reduz risco de não atingir a meta):")
        print(coefs.head(8).round(3).to_string())
        print(coefs.tail(8).round(3).to_string())
        coefs.to_csv(REPORTS / "coeficientes_logistica.csv")

    pred = df.copy()
    pred["proba_nao_atingiu"] = melhor_pipe.predict_proba(df[NUM_FEATURES + CAT_FEATURES])[:, 1]
    ranking = pred.sort_values("proba_nao_atingiu", ascending=False)[
        ["id_municipio", "sigla_uf", "regiao", "taxa_alfabetizacao", "nao_atingiu_meta", "proba_nao_atingiu"]
    ]
    ranking.head(100).to_csv(REPORTS / "ranking_risco_predito_top100.csv", index=False)
    print("artefatos em reports/ e images/modelagem/")


if __name__ == "__main__":
    main()
