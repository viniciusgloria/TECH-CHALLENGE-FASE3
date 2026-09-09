# Predição e inteligência analítica para alfabetização no Brasil

Tech Challenge Fase 3 | FIAP Pós Tech AI Scientist

Autor: Vinícius Gomes Machado Gloria

Modelo supervisionado de classificação binária para antecipar municípios em risco de não cumprir a meta do Compromisso Nacional Criança Alfabetizada, com interpretabilidade (SHAP) e leitura para política pública.

Vídeo executivo (até 5 minutos): https://www.loom.com/share/430e038a03574289950bfbe3a2f643db

Slides: https://docs.google.com/presentation/d/1oJDTPyb5I2uWLme_pxXXtbShCo6L3oTHbZwdZrxXI8A/edit?usp=sharing

Roteiro: `docs/roteiro_video.md`

## Contexto do problema

Alfabetizar na idade certa é uma das metas centrais da política educacional brasileira até 2030. O Indicador Criança Alfabetizada mede a proporção de crianças do 2º ano do ensino fundamental da rede municipal que atingem o ponto de corte da Pesquisa Alfabetiza Brasil.

Olhar só o nível da taxa não basta para decidir. A meta municipal é pactuada por trajetória: um município com 70% pode estar acima ou abaixo da barra, conforme o que foi combinado para aquele ano. Gestores precisam saber quem tende a não cumprir a meta, quais padrões regionais se repetem e quais fatores pesam na previsão, sem tratar associação como causa.

Este projeto continua a camada Gold da Fase 2 (engenharia de dados do mesmo indicador) e entra na etapa de Modeling do CRISP-DM.

## Objetivo analítico

Prever, no recorte de 2024, se o município **não atingirá a meta** do ano (`nao_atingiu_meta = 1`).

A unidade de análise é o município, não o aluno. O enunciado pede classificação de aluno, mas a Gold nacional não tem microdado nacional. Essa escolha é declarada aqui de propósito: o rótulo binário honesto que a Fase 2 já calcula é o cumprimento da meta municipal.

Classe positiva: risco educacional (não bateu a meta). Recorte: 5.232 municípios com meta preenchida em 2024 (46,7% positivos). 2023 entra só como lag.

Perguntas de negócio que o projeto responde:

1. Quais fatores mais impactam a previsão de cumprimento da meta?
2. Quais municípios apresentam maior risco?
3. Quais regiões possuem padrões semelhantes?
4. Como antecipar quem pode não atingir metas de 2025 a 2030?
5. Quais variáveis mais influenciam o modelo?

## Descrição da base utilizada

Fonte primária: dataset público `basedosdados.br_inep_avaliacao_alfabetizacao` (mesma origem da Fase 2). A Gold usada no modelo contém:

- Indicador Criança Alfabetizada por município (rede Municipal)
- Metas municipais 2024 a 2030 (alvo: cumprir a meta de 2024)
- Metas estaduais 2024 (`meta_uf_ano_corrente`, feature)
- Meta nacional 2024 (59,9; constante, registrada na base e fora do `X`)
- Território (UF e região a partir do código IBGE)
- Indicador temporal (taxa de 2023)
- População e PIB (IBGE 2023)
- Dados educacionais complementares (Censo Escolar 2023)
- Socioeconômico estrutural: IDHM 2010 (Atlas)

Consultas em `src/ingestao/consultas_enriquecimento.sql`. Base de modelagem: `data/silver/base_modelo.parquet`.

Fora do modelo, por vazamento: taxa de 2024, média de português, gap e variação do mesmo ano.

## Etapas de modelagem

1. Reconstrução da Gold da Fase 2 em pandas (`src/preprocessing/reconstruir_gold.py`).
2. EDA e definição do alvo (`src/evaluation/eda.py`, `reports/01_eda_e_definicao_alvo.md`).
3. Clusterização para padrões semelhantes (`src/modeling/clusterizacao.py`).
4. Risco de metas futuras por persistência e tendência (`src/evaluation/risco_metas_futuras.py`).
5. Join do enriquecimento (`src/preprocessing/montar_base_modelo.py`).
6. Pipeline Scikit-learn: `ColumnTransformer` com imputação, scaling e one-hot, integrado ao estimador; `train_test_split` estratificado; `GridSearchCV` com `StratifiedKFold` (`src/modeling/treinar.py`).
7. Interpretação com SHAP (e importância das variáveis no ranking SHAP).

## Escolha do algoritmo

Candidatos: dummy da classe majoritária, regressão logística, Random Forest e HistGradientBoosting. O **HistGradientBoosting** (`learning_rate=0,05`, `max_depth=6`) é o modelo oficial: melhor F1 no teste (0,678) e na validação cruzada. A logística fica logo atrás (F1 0,671).

## Métricas de avaliação

Foco na classe de risco. Acurácia sozinha engana (dummy acerta 53% e F1 0). Métricas no teste (1.047 municípios):

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0,533 | 0,000 | 0,000 | 0,000 | 0,500 | 0,467 |
| Logística | 0,710 | 0,713 | 0,634 | 0,671 | 0,794 | 0,778 |
| Random Forest | 0,707 | 0,715 | 0,620 | 0,664 | 0,799 | 0,782 |
| HistGradientBoosting | 0,714 | 0,713 | 0,646 | 0,678 | 0,797 | 0,781 |

Recall 0,65: de cada 100 municípios que não bateram a meta, o modelo aponta 65. O falso negativo é o erro caro para priorização orçamentária.

## Interpretação dos resultados

Cumprir a meta não é a mesma coisa que ter taxa alta. O Sul tem taxa média 64,7 e só 34% na meta, porque a barra média é 72,3. O Centro-Oeste tem taxa 72,3 e 73% na meta. Rio Grande do Sul: 9,5% na meta. Ceará: 91%.

A meta de 2024 correlaciona 0,979 com a taxa de 2023. Com a meta estadual no modelo, o SHAP aponta ela como o fator de maior impacto, depois a taxa de 2023, RS e porte. PIB per capita continua residual. A meta nacional (59,9) é a mesma para todos os municípios e por isso não entra no classificador.

Associação preditiva, não efeito causal (Aula 8 de Supervisionados).

## Insights encontrados

- Política que ranqueie só pela taxa ignora o RS (barra alta) e superestima o Norte (barra mais baixa).
- K-Means (K=6) reproduz as grandes regiões, com uma quebra só no Nordeste: grupo de taxa 43% versus grupo de 77%.
- Se a taxa de 2024 não subir, 56% dos municípios ficam abaixo da meta de 2025 e 79% abaixo da de 2030. Se repetirem o ritmo 2023-2024, cerca de metade permanece em risco.
- Ranking preditivo: `reports/ranking_risco_predito_top100.csv`.

## Limitações do projeto

- O modelo classifica município, não aluno.
- Split aleatório estratificado, sem `GroupKFold` por UF neste treino.
- IDHM é de 2010. Censo 2023 agrega todas as redes, não só a municipal.
- Componentes setoriais do PIB 2023 vieram nulos na extração.
- FUNDEB, PNAD e Cadastro Único não foram usados.
- AC, RR e DF não entram no recorte com alvo em 2024.
- Aprendizagem por reforço ficou fora: não há coluna de ação nem de recompensa na Gold.

## Aplicação prática para políticas públicas

O ranking de probabilidade de não cumprir a meta, cruzado com o gap observado, vira fila de priorização: onde a trajetória está falhando, não apenas onde a taxa é baixa. Ceará e Minas sugerem que a barra pode ser superada; Bahia e RS concentram risco de trajetória e pedem diagnóstico local (rede, formação, frequência), não só recurso proporcional à pobreza.

As metas futuras mostram que persistir no nível de 2024 não fecha 2030. A velocidade recente segura cerca de metade dos municípios; a outra metade precisa acelerar.

## Possíveis evoluções futuras

- Validação agrupada por UF e monitoramento de drift.
- FUNDEB e indicadores de insumos (docentes, jornada).
- Modelo de regressão da taxa, paralelo ao classificador de meta, para separar nível e trajetória.
- RL para alocação de recursos só se existirem histórico de intervenção, recompensa e auditoria de cobertura. Hoje essas pré-condições não existem.

## Como reproduzir

Python 3.11. CSVs em `data/raw/` (não versionados). Ambiente:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Ordem:

```
python src/preprocessing/reconstruir_gold.py
python src/evaluation/eda.py
python src/modeling/clusterizacao.py
python src/evaluation/risco_metas_futuras.py
python src/preprocessing/montar_base_modelo.py
python src/modeling/treinar.py
```

Notebooks em `notebooks/`. Relatórios em `reports/`. Figuras em `images/`.

## Estrutura

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/raw, data/silver, data/gold
├── notebooks/
├── src/preprocessing, modeling, evaluation, visualization, ingestao
├── reports/
├── images/
└── docs/
```
