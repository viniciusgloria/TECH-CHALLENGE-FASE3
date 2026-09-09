# Caminho para cumprir o Tech Challenge da Fase 3

O enunciado pede um modelo supervisionado, pipeline Scikit-learn, interpretabilidade e cinco perguntas de negócio. O alvo formal está em `reports/01_eda_e_definicao_alvo.md`: município em 2024, classe positiva `nao_atingiu_meta`.

## Cobertura das perguntas

| Pergunta do enunciado | Como responder | Status |
|---|---|---|
| Quais fatores mais impactam a alfabetização? | SHAP no HistGradientBoosting; meta estadual, taxa 2023, RS e porte | Feito |
| Quais municípios apresentam maior risco educacional? | `reports/ranking_risco_predito_top100.csv` | Feito |
| Quais regiões possuem padrões semelhantes? | K-Means K=6, Silhouette | Feito |
| Como prever municípios que podem não atingir metas futuras? | Persistência e tendência vs metas 2025-2030 | Feito |
| Quais variáveis possuem maior influência nos modelos? | SHAP | Feito |

FUNDEB, PNAD e Cadastro Único ficaram de fora de propósito (opcionais no PDF).
