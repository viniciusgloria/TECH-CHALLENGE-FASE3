# Modelagem supervisionada

Base: `data/silver/base_modelo.parquet` (5.232 municípios de 2024 com meta). Alvo: `nao_atingiu_meta` (46,7% positivo). Script: `src/modeling/treinar.py`.

A Gold inclui meta municipal (alvo), meta estadual (`meta_uf_ano_corrente` como feature) e meta nacional (valor único de 2024, 59,9; não entra no `X` porque é constante entre municípios).

## Pipeline

`ColumnTransformer` dentro de `Pipeline` do Scikit-learn, ajustado só no treino (`train_test_split` 80/20 estratificado, `random_state=42`, `StratifiedKFold` 5 folds).

Numéricos (`SimpleImputer` mediana + `StandardScaler`): taxa de 2023, log da população, PIB per capita, escolas por mil habitantes, infraestrutura do Censo 2023, IDHM educação/longevidade/renda (2010) e meta estadual de 2024.

Categóricos (`SimpleImputer` mais frequente + `OneHotEncoder` com `handle_unknown='ignore'`): UF e região.

Fora do `X`, por vazamento: taxa de 2024, média de português, gap e variação do mesmo ano.

## Comparação no teste (1.047 municípios)

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | PR-AUC | F1 na CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (mais frequente) | 0,533 | 0,000 | 0,000 | 0,000 | 0,500 | 0,467 | - |
| Regressão logística | 0,710 | 0,713 | 0,634 | 0,671 | 0,794 | 0,778 | 0,650 |
| Random Forest | 0,707 | 0,715 | 0,620 | 0,664 | 0,799 | 0,782 | 0,640 |
| HistGradientBoosting | 0,714 | 0,713 | 0,646 | 0,678 | 0,797 | 0,781 | 0,655 |

Modelo oficial: HistGradientBoosting (`learning_rate=0,05`, `max_depth=6`). Melhor F1 no teste e na CV. Recall 0,65 na classe de risco.

## O que o SHAP diz

Com a meta estadual no modelo, ela passa a ser o fator de maior impacto médio absoluto, seguida da taxa de 2023, UF (RS), porte populacional e Sul. PIB per capita continua residual.

Leitura: a barra do estado ajuda a prever se o município cumpre a trajetória local. Não substitui UF (RS segue com peso próprio). Associação preditiva, não causalidade.

## Ranking de risco predito

`reports/ranking_risco_predito_top100.csv`.

## Limites

Split aleatório estratificado, sem `GroupKFold` por UF. IDHM de 2010. Censo com todas as redes. Meta nacional é a mesma para todos os municípios. O modelo classifica município, não aluno.
