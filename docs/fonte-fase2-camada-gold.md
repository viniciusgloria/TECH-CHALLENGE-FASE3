# Inventário da camada Gold da Fase 2

Fonte: código e documentação em `FASE 2/TECH CHALLENGE`. Nenhum dado tabular da Gold está versionado neste workspace. Os dados vivem no Databricks (catálogo `workspace`, schema `alfabetizacao_gold`) e a origem é o dataset público `basedosdados.br_inep_avaliacao_alfabetizacao` no BigQuery.

## O que a Fase 2 realmente entregou

Pipeline medalhão (Bronze, Silver, Gold) no Databricks Free Edition, com extração batch do BigQuery e streaming simulado via Auto Loader. A unidade de análise da base analítica é **município-ano**, não aluno.

A tabela central, lida pela Gold, é a Silver `municipio_ano_integrada`. A Gold não cria features novas: ela recorta e agrega essa tabela em quatro produtos de consumo.

## Tabelas Gold

Geradas por `src/gold/03_carga_gold.py`, particionadas por `ano`.

### 1. `indicador_por_municipio`

Foto da medição mais recente de cada município (window `row_number` por `id_municipio` ordenado por `ano` desc). Herda o schema completo de `municipio_ano_integrada`.

### 2. `meta_versus_resultado`

Recorte de linhas com meta preenchida. Colunas: `ano`, `id_municipio`, `sigla_uf`, `regiao`, `taxa_alfabetizacao`, `meta_ano_corrente`, `gap_meta`, `atingiu_meta`.

### 3. `evolucao_temporal`

Série por município: `taxa_alfabetizacao`, `taxa_ano_anterior` (`lag`) e `variacao`. Cobertura temporal curta (comentários da própria Fase 2: indicador observado em 2023 e 2024).

### 4. `resumo_por_uf`

Agregado executivo: quantidade de municípios, taxa média e proporção de municípios na meta, por UF, região e ano.

## Schema da base analítica (`municipio_ano_integrada`)

| Coluna | Origem | Tipo efetivo | Papel |
|---|---|---|---|
| `ano` | indicador municipal | int | chave temporal |
| `id_municipio` | código IBGE 7 dígitos (string) | string | chave de entidade |
| `sigla_uf`, `nome_uf`, `regiao` | mapa fixo dos 2 primeiros dígitos do IBGE | string | território |
| `taxa_alfabetizacao` | tabela `municipio`, rede Municipal (`rede = 3`) | double | indicador |
| `media_portugues` | mesma tabela de indicador | double | contexto da avaliação |
| `meta_ano_corrente` | meta municipal despivotada (2024 a 2030) | double | meta pactuada |
| `gap_meta` | `taxa_alfabetizacao - meta_ano_corrente` | double | derivado |
| `atingiu_meta` | `1` se gap >= 0, senão `0` | int | alvo binário candidato |

## O que a Fase 2 não colocou na Gold

O enunciado da Fase 3 cita dados territoriais, socioeconômicos, educacionais complementares, indicadores temporais e informações populacionais. Na Gold construída, o território se resume a UF e região. Não há IBGE socioeconômico, Censo Escolar, FUNDEB, PNAD, Atlas nem Cadastro Único. O próprio `docs/arquitetura.md` da Fase 2 registra isso como evolução futura.

A tabela `alunos` existe na Bronze e gera `indicador_alunos_municipio` na Silver, mas é amostra: 20 mil linhas de municípios de São Paulo (`id_municipio LIKE '35%'`). A Fase 2 declara explicitamente que a base analítica usa a taxa já agregada do município, não o microdado.

## Decisões da Fase 2 que o modelo da Fase 3 herda

- Indicador municipal filtrado em rede Municipal (`rede = 3`); indicador de UF em rede Pública Estadual e Municipal (`rede = 5`). O corte Total (`rede = 0`) foi descartado por cobertura quase nula.
- Join de indicador com meta usa apenas `(ano, id_municipio)`, sem cruzar a coluna `rede`, porque as metas já estão na rede correspondente em texto.
- Metas entram em formato longo (uma linha por ano de 2024 a 2030). O join é `ano do indicador = ano_meta`. Anos de indicador anteriores a 2024 ficam sem meta, e a Gold `meta_versus_resultado` descarta essas linhas.
- `id_municipio` é string, para não perder zero à esquerda.
- Qualidade valida unicidade da chave `(ano, id_municipio)`, nulos de chave (tolerância zero), nulos de taxa (teto 10%) e taxa no intervalo `[0, 100]`.

## Inconsistências internas da Fase 2 (relevantes para a Fase 3)

1. Escala da taxa. O script de qualidade exige intervalo `[0, 100]`. O evento de exemplo em `data/sample/evento_exemplo.json` traz `indicador_alfabetizacao: 0.812`. Antes de modelar, é obrigatório conferir se a Gold gravou proporção (0 a 1) ou percentual (0 a 100).
2. `docs/arquitetura.md` afirma que o indicador deriva do corte de 743 pontos do Saeb sobre alunos. O código não faz isso. A taxa entra pronta da tabela agregada `municipio`. A tabela de alunos é apenas demonstração de ingestão.
3. `media_portugues` vem da mesma avaliação que gera `taxa_alfabetizacao`. Usar essa coluna como feature para predizer alfabetização é risco direto de data leakage.

## Implicações para o Tech Challenge da Fase 3

O enunciado pede prever se um aluno será alfabetizado ou não. A Gold disponível não tem aluno nacional. O grão honesto é município-ano.

Alvo binário possível sem inventar rótulo: `atingiu_meta` (município atingiu ou não a meta do ano). Alternativa: discretizar `taxa_alfabetizacao` contra um corte. As duas opções predizem município, não aluno. Isso precisa ficar declarado no README.

Features usáveis hoje, sem enriquecimento: `sigla_uf`, `regiao`, possivelmente `ano`, e as séries de `evolucao_temporal` (`taxa_ano_anterior`, `variacao`) se a validação for temporal. `taxa_alfabetizacao`, `gap_meta` e `atingiu_meta` não podem coexistir como feature e alvo. `media_portugues` deve ser tratada como suspeita de vazamento até prova em contrário.

## Situação após reconstrução local

Os CSVs do BigQuery foram gravados em `data/raw/` (`indicadormunicipal.csv` e `metamunicipal.csv`). O script `src/preprocessing/reconstruir_gold.py` replicou Silver e Gold em parquet.

Números observados:

- Indicador: 10.896 linhas, 5.448 por ano (2023 e 2024), 5.500 municípios, rede Municipal, 2º ano, taxa na escala 0 a 100 (mín. 4,35, máx. 100, média 61,54). Zero nulos.
- Meta: 5.352 municípios com ano-base 2023 e 2024. Deduplicação usa o ano-base mais recente, como na Fase 2.
- Join `ano do indicador = ano da meta`: 2023 fica inteiro sem meta (5.448 nulos). Em 2024, 216 municípios sem meta. Alvo binário `atingiu_meta` existe em 5.232 linhas, todas de 2024: 2.788 atingiram a meta e 2.444 não atingiram.

O modelo supervisionado da Fase 3, se usar `atingiu_meta`, é um recorte transversal de 2024, não um painel de dois anos. 2023 só entra como lag em `evolucao_temporal`.
