# Predição e inteligência analítica para alfabetização no Brasil

Tech Challenge Fase 3, FIAP Pós Tech AI Scientist.

Vinícius Gomes Machado Gloria

Classificador de município em 2024: quem tende a não cumprir a meta do Compromisso Nacional Criança Alfabetizada.

Vídeo (até 5 minutos): https://www.loom.com/share/430e038a03574289950bfbe3a2f643db

Slides: https://docs.google.com/presentation/d/1oJDTPyb5I2uWLme_pxXXtbShCo6L3oTHbZwdZrxXI8A/edit?usp=sharing

## Contexto do problema

O Brasil se comprometeu a alfabetizar na idade certa até 2030. O Indicador Criança Alfabetizada mede a proporção de crianças do 2º ano da rede municipal que passam no corte da Pesquisa Alfabetiza Brasil.

Olhar só a taxa do município não fecha a decisão. A meta é pactuada por trajetória. Um município com 70% pode estar bem ou mal, depende da barra daquele ano. A pergunta útil para a gestão é quem tende a ficar abaixo da meta.

Este trabalho parte da Gold da Fase 2 e entra na modelagem.

## Objetivo analítico

No recorte de 2024, prever se o município **não atinge a meta** (`nao_atingiu_meta = 1`).

O enunciado pede aluno. A Gold nacional não tem microdado. O rótulo que dá para usar com honestidade é o cumprimento da meta municipal. São 5.232 municípios com meta preenchida (46,7% positivos). 2023 entra só como taxa do ano anterior.

Perguntas que o projeto responde:

1. Quais fatores mais pesam na previsão de cumprimento da meta?
2. Quais municípios têm maior risco?
3. Quais regiões se parecem?
4. Como antecipar quem pode não bater as metas de 2025 a 2030?
5. Quais variáveis mais influenciam o modelo?

## Descrição da base utilizada

Fonte: `basedosdados.br_inep_avaliacao_alfabetizacao`, a mesma da Fase 2. No modelo entram:

- Indicador Criança Alfabetizada (rede municipal)
- Metas municipais 2024 a 2030 (o alvo é a meta de 2024)
- Meta estadual de 2024 (`meta_uf_ano_corrente`)
- Meta nacional de 2024 (59,9; constante, fica na base e fora do `X`)
- UF e região
- Taxa de 2023
- População e PIB (IBGE 2023)
- Censo Escolar 2023
- IDHM 2010 (Atlas)

Consultas em `src/ingestao/consultas_enriquecimento.sql`. Base de treino: `data/silver/base_modelo.parquet`.

Fora do `X`, por vazamento: taxa de 2024, média de português, gap e variação do mesmo ano.

## Etapas de modelagem

1. Reconstruir a Gold em pandas (`src/preprocessing/reconstruir_gold.py`).
2. EDA e definição do alvo (`src/evaluation/eda.py`, `reports/01_eda_e_definicao_alvo.md`).
3. Clusterização (`src/modeling/clusterizacao.py`).
4. Risco das metas 2025 a 2030 (`src/evaluation/risco_metas_futuras.py`).
5. Join do enriquecimento (`src/preprocessing/montar_base_modelo.py`).
6. Pipeline Scikit-learn com imputação, scaling, one-hot, `train_test_split` estratificado e `GridSearchCV` (`src/modeling/treinar.py`).
7. SHAP.

## Escolha do algoritmo

Comparei dummy da classe majoritária, regressão logística, Random Forest e HistGradientBoosting. Fiquei com o **HistGradientBoosting** (`learning_rate=0,05`, `max_depth=6`): melhor F1 no teste (0,678) e na validação cruzada. A logística ficou em 0,671.

## Métricas de avaliação

Acurácia sozinha engana. O dummy acerta 53% e não acha nenhum município de risco (F1 0). Teste com 1.047 municípios:

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0,533 | 0,000 | 0,000 | 0,000 | 0,500 | 0,467 |
| Logística | 0,710 | 0,713 | 0,634 | 0,671 | 0,794 | 0,778 |
| Random Forest | 0,707 | 0,715 | 0,620 | 0,664 | 0,799 | 0,782 |
| HistGradientBoosting | 0,714 | 0,713 | 0,646 | 0,678 | 0,797 | 0,781 |

Recall 0,65: de cada 100 municípios que não bateram a meta, o modelo aponta 65. Esse é o erro que mais custa se a fila de priorização errar.

## Interpretação dos resultados

Cumprir a meta não é ter taxa alta. O Sul tem taxa média 64,7 e só 34% na meta, porque a barra média é 72,3. O Centro-Oeste tem taxa 72,3 e 73% na meta. Rio Grande do Sul: 9,5% na meta. Ceará: 91%.

A meta de 2024 correlaciona 0,979 com a taxa de 2023. No SHAP, o que mais pesa é a meta estadual, depois a taxa de 2023, o Rio Grande do Sul e o porte. PIB per capita quase não entra. A meta nacional é 59,9 para todo mundo, então não separa município.

Isso é associação, não causa.

## Insights encontrados

- Ranquear só pela taxa esconde o RS e aponta demais para o Norte.
- K-Means com K=6 devolve as grandes regiões. A quebra extra é no Nordeste: um grupo com taxa 43% e outro com 77%.
- Se a taxa de 2024 não subir, 56% ficam abaixo da meta de 2025 e 79% abaixo da de 2030. Se repetir o ritmo 2023-2024, cerca de metade continua em risco.
- Ranking: `reports/ranking_risco_predito_top100.csv`.

## Limitações do projeto

- Classifica município, não aluno.
- Split estratificado, sem `GroupKFold` por UF.
- IDHM de 2010. Censo 2023 mistura redes, não só a municipal.
- Componentes setoriais do PIB 2023 vieram nulos.
- Não usei FUNDEB, PNAD nem Cadastro Único.
- AC, RR e DF não entram no recorte com alvo em 2024.
- Sem aprendizagem por reforço: a Gold não tem ação nem recompensa.

## Aplicação prática para políticas públicas

O ranking de probabilidade de não cumprir, cruzado com o gap, vira fila de trabalho. Onde a barra é alta e o resultado não anda, a conversa não é só mandar recurso para quem é mais pobre.

Ceará e Minas mostram que a barra dá para passar. Bahia e RS pedem visita e diagnóstico local. Manter o nível de 2024 não fecha 2030.

## Possíveis evoluções futuras

- Validação por UF.
- FUNDEB e insumos da escola.
- Um modelo da taxa, paralelo ao classificador de meta.

## Como reproduzir

Python 3.11. Os CSVs ficam em `data/raw/` e não vão para o Git.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

```
python src/preprocessing/reconstruir_gold.py
python src/evaluation/eda.py
python src/modeling/clusterizacao.py
python src/evaluation/risco_metas_futuras.py
python src/preprocessing/montar_base_modelo.py
python src/modeling/treinar.py
```

Notebooks em `notebooks/`. Relatórios em `reports/`. Figuras em `images/`.
