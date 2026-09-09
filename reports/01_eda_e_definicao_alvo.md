# EDA e definição do alvo

Fonte: `data/silver/municipio_ano_integrada.parquet` e `data/gold/evolucao_temporal.parquet`, reconstruídos a partir da Gold da Fase 2. Script: `src/evaluation/eda.py`. Figuras em `images/eda/`.

## Qualidade da base

10.896 linhas, chave única `(ano, id_municipio)`, 5.500 municípios. Anos 2023 e 2024. UF e região sem nulos. Taxa e média de português sem nulos e taxa dentro de `[0, 100]`.

`meta_ano_corrente`, `gap_meta` e `atingiu_meta` estão ausentes em 51,98% das linhas. Isso não é falha de ingestão: o join da Fase 2 casa o ano do indicador com o ano da meta, e as metas só existem de 2024 a 2030. Em 2023 as 5.448 linhas ficam sem alvo. Em 2024, 216 municípios (taxa média 51,5, abaixo da média nacional) também ficam sem meta.

104 municípios aparecem em apenas um dos dois anos.

UFs ausentes no recorte com alvo em 2024: AC, RR e DF. A base de modelagem cobre 24 UFs.

## O que a distribuição mostra

A taxa média sobe de 60,28 em 2023 para 62,80 em 2024. O histograma de 2024 é largo (desvio 19,3), com mediana 64,05. Não há um pico único: o país é heterogêneo.

A comparação taxa versus meta em 2024 mostra a diagonal `taxa = meta` como fronteira do rótulo. Há um teto visível de meta em 80%, herdado da trajetória pactuada até 2030. Municípios acima da diagonal atingiram a meta; abaixo, não.

## Geografia: nível alto não é a mesma coisa que bater a meta

Proporção de municípios na meta em 2024:

| Região | n | Taxa média | Meta média | Prop. na meta |
|---|---:|---:|---:|---:|
| Sul | 1054 | 64,68 | 72,31 | 0,34 |
| Norte | 389 | 50,24 | 50,95 | 0,44 |
| Nordeste | 1738 | 56,78 | 55,67 | 0,50 |
| Sudeste | 1590 | 69,98 | 64,23 | 0,66 |
| Centro-Oeste | 461 | 72,32 | 65,63 | 0,73 |

O Sul tem taxa acima da média nacional e a pior adesão à meta, porque a barra é a mais alta (72,3). O Norte tem a menor taxa e ainda assim adesão melhor que o Sul, porque a meta média é 51,0. Ceará é o extremo oposto: taxa 90,3 e 91,3% dos municípios na meta. Rio Grande do Sul é o extremo de risco: taxa 53,1 e só 9,5% na meta.

Hipótese 1: `atingiu_meta` mede cumprimento de trajetória, não alfabetização absoluta. Política pública que ranqueie só pela taxa vai priorizar o Norte e a Bahia e ignorar o RS, onde a meta é mais exigente.

## Vazamentos e colinearidades

Pearson no recorte 2024 com alvo:

- `taxa_alfabetizacao` vs `media_portugues`: 0,922 (0,930 em 2023)
- `meta_ano_corrente` vs `taxa_ano_anterior`: 0,979
- `gap_meta` vs `variacao`: 0,936

Hipótese 2: `media_portugues` é a mesma avaliação que gera a taxa. Entrar como feature para prever alfabetização (ou o cumprimento da meta derivada da taxa) é data leakage.

Hipótese 3: a meta de 2024 é, na prática, uma função da taxa de 2023 com teto 80. Por isso `meta_ano_corrente` e `taxa_ano_anterior` são quase a mesma informação. Usar as duas juntas infla colinearidade sem ganho.

`variacao` usa a taxa de 2024. Se o alvo é função da taxa de 2024, `variacao` vaza o rótulo.

## Baseline ingênua

Regra: o município atinge a meta de 2024 se a taxa de 2023 já era maior ou igual a essa meta. Cobertura do lag: 100% no recorte com alvo. Acurácia: 0,485, abaixo da classe majoritária (0,533). A regra marca positivo em só 17,8% dos casos, porque a meta de 2024 em geral fica acima da taxa de 2023.

Hipótese 4: bater a meta de 2024 exige avanços em relação a 2023, não manutenção. Um modelo útil precisa superar tanto a classe majoritária quanto essa regra de inércia.

## Definição formal do alvo

O enunciado pede prever se um aluno será alfabetizado. A Gold nacional não tem aluno. A unidade honesta é o município.

Decisão:

- Unidade de análise: município
- Recorte: ano 2024, linhas com meta preenchida (5.232 municípios)
- Variável alvo: `nao_atingiu_meta = 1` se `atingiu_meta == 0`, e `0` caso contrário
- Classe positiva: município que não atingiu a meta (risco educacional)
- Balanceamento: 2.444 positivos (46,7%) e 2.788 negativos (53,3%)

Essa escolha atende o enunciado nas perguntas de risco e de metas futuras, permanece classificação binária, e evita inventar um corte arbitrário sobre a taxa. O README declara que o modelo não classifica aluno.

## Features permitidas neste recorte, sem enriquecimento

Permitidas: `sigla_uf`, `regiao`, `taxa_ano_anterior`. `meta_ano_corrente` é redundante com o lag e fica de fora do conjunto inicial.

Proibidas: `taxa_alfabetizacao` do mesmo ano, `media_portugues` do mesmo ano, `gap_meta`, `variacao`, `atingiu_meta`.

Com só território e lag, o modelo descreve inércia geográfica. Isso é insuficiente para as perguntas de fatores socioeconômicos do enunciado. O salto seguinte foi o enriquecimento da base (IBGE, Censo Escolar, Atlas), que o enunciado autoriza e a Fase 2 não fez.

## Hipóteses analíticas que a modelagem deve testar

1. Cumprir a meta depende da distância entre a taxa de 2023 e a barra de 2024, mais o contexto regional, não do nível absoluto de alfabetização.
2. Municípios do Sul com taxa relativamente alta podem ser de alto risco porque a meta é mais exigente.
3. Sem features socioeconômicas, a performance fora da UF de treino cai, o que exige validação agrupada por UF, não um `train_test_split` aleatório.
4. Interpretação SHAP de renda ou de território, quando essas features existirem, será associação preditiva, não efeito causal (Aula 8 de Supervisionados).
