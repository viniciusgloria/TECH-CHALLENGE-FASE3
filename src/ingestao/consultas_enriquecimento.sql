-- Enriquecimento da base analítica da Fase 3.
-- Mesma plataforma da Fase 2: projeto basedosdados no BigQuery sandbox.
-- Projeto de cobrança: tech-challenge-fase2-502418
--
-- Rodar uma consulta por vez, conferir o total de linhas (não a amostra da tela)
-- e salvar CSV em FASE 3/TECH CHALLENGE/data/raw/ com o nome indicado.
-- Features de 2023 ou anteriores: não usam o resultado de alfabetização de 2024.

-- ============================================================
-- 1. ibge_pib.csv
-- PIB municipal a preços correntes. Schema confirmado no modelo
-- basedosdados/pipelines: pib, impostos_liquidos, va e componentes.
-- Ano mais recente até 2023, para não vazar o ciclo de 2024.
-- ============================================================
SELECT
  ano,
  id_municipio,
  pib,
  impostos_liquidos,
  va,
  va_agropecuaria,
  va_industria,
  va_servicos,
  va_adespss
FROM `basedosdados.br_ibge_pib.municipio`
WHERE ano = (
  SELECT MAX(ano)
  FROM `basedosdados.br_ibge_pib.municipio`
  WHERE ano <= 2023
);

-- ============================================================
-- 2. ibge_populacao.csv
-- Estimativa de população. Serve para PIB per capita e porte municipal.
-- ============================================================
SELECT
  ano,
  id_municipio,
  sigla_uf,
  populacao
FROM `basedosdados.br_ibge_populacao.municipio`
WHERE ano = 2023;

-- ============================================================
-- 3. censo_escola_mun.csv
-- Censo Escolar 2023 agregado no município, rede municipal.
-- agua_potavel aparece no material da Base dos Dados; se alguma coluna
-- de infraestrutura falhar, rode só id_municipio + COUNT(*) e avise.
-- ============================================================
SELECT
  id_municipio,
  COUNT(*) AS qtd_escolas,
  AVG(IF(CAST(agua_potavel AS STRING) IN ('1', 'true', 'Sim'), 1, 0)) AS prop_agua_potavel,
  AVG(IF(CAST(internet AS STRING) IN ('1', 'true', 'Sim'), 1, 0)) AS prop_internet,
  AVG(IF(CAST(banda_larga AS STRING) IN ('1', 'true', 'Sim'), 1, 0)) AS prop_banda_larga,
  AVG(IF(CAST(biblioteca AS STRING) IN ('1', 'true', 'Sim'), 1, 0)) AS prop_biblioteca,
  AVG(IF(CAST(laboratorio_informatica AS STRING) IN ('1', 'true', 'Sim'), 1, 0)) AS prop_lab_informatica
FROM `basedosdados.br_inep_censo_escolar.escola`
WHERE ano = 2023
GROUP BY id_municipio;

-- ============================================================
-- 4. atlas_idhm.csv
-- Atlas do Desenvolvimento Humano, recorte 2010 (último ano municipal
-- da série clássica). Declarar defasagem temporal no README.
-- Se o dataset não aparecer com este nome, busque "Atlas" ou "IDHM"
-- no explorador do projeto basedosdados e ajuste o FROM.
-- ============================================================
SELECT
  ano,
  id_municipio,
  idhm,
  idhm_e,
  idhm_l,
  idhm_r
FROM `basedosdados.mundo_onu_adh.municipio`
WHERE ano = 2010;


-- ============================================================
-- 5. meta_uf.csv
-- ============================================================
SELECT
  ano,
  sigla_uf,
  rede,
  taxa_alfabetizacao,
  meta_alfabetizacao_2024,
  meta_alfabetizacao_2025,
  meta_alfabetizacao_2026,
  meta_alfabetizacao_2027,
  meta_alfabetizacao_2028,
  meta_alfabetizacao_2029,
  meta_alfabetizacao_2030,
  percentual_participacao
FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_uf`;


-- ============================================================
-- 6. meta_brasil.csv
-- ============================================================
SELECT
  ano,
  rede,
  taxa_alfabetizacao,
  meta_alfabetizacao_2024,
  meta_alfabetizacao_2025,
  meta_alfabetizacao_2026,
  meta_alfabetizacao_2027,
  meta_alfabetizacao_2028,
  meta_alfabetizacao_2029,
  meta_alfabetizacao_2030,
  percentual_participacao
FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_brasil`;
