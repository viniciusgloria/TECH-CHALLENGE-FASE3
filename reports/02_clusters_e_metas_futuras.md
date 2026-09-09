# Clusters e risco de metas futuras

Scripts: `src/modeling/clusterizacao.py` e `src/evaluation/risco_metas_futuras.py`.

## Padrões semelhantes (clusterização)

K-Means com `StandardScaler`, `random_state=42`, features `taxa_alfabetizacao`, `taxa_ano_anterior`, `meta_ano_corrente` e dummies de região. Silhouette máximo em K=6 (0,555).

O mix regional mostra que cinco dos seis clusters coincidem com as grandes regiões. A única quebra intra-regional é o Nordeste:

| Cluster | n | Taxa 2024 | Meta | Prop. não atingiu | Região dominante |
|---|---:|---:|---:|---:|---|
| 0 | 389 | 50,2 | 51,0 | 0,56 | Norte |
| 1 | 1050 | 64,8 | 72,5 | 0,66 | Sul |
| 2 | 1064 | 43,3 | 44,2 | 0,59 | Nordeste (baixa taxa) |
| 3 | 682 | 77,6 | 73,3 | 0,35 | Nordeste (alta taxa) |
| 4 | 461 | 72,3 | 65,6 | 0,27 | Centro-Oeste |
| 5 | 1586 | 70,1 | 64,3 | 0,34 | Sudeste |

Leitura: sem features socioeconômicas, "regiões semelhantes" reproduz o recorte geográfico, com um segundo eixo só no Nordeste (Ceará e pares no cluster 3 versus Bahia e pares no cluster 2). Isso confirma a EDA.

## Municípios que podem não atingir metas futuras

Metas 2025 a 2030 já estavam na Gold (`metamunicipal.csv`). Dois cenários:

- Persistência: a taxa de 2024 se mantém.
- Tendência: o município repete, a cada ano, o incremento observado de 2023 para 2024.

| Ano | Risco persistência | Risco tendência | Meta média |
|---|---:|---:|---:|
| 2025 | 0,559 | 0,484 | 65,4 |
| 2026 | 0,627 | 0,492 | 68,7 |
| 2027 | 0,686 | 0,495 | 71,9 |
| 2028 | 0,731 | 0,496 | 74,9 |
| 2029 | 0,762 | 0,494 | 77,6 |
| 2030 | 0,785 | 0,494 | 80,0 |

Se ninguém melhorar, 56% já ficam abaixo da meta de 2025 e 79% abaixo da de 2030 (teto 80). Se a velocidade de 2023-2024 se repetir, cerca de metade permanece em risco em qualquer horizonte: a barra sobe quase no mesmo ritmo do avanço recente.

Risco de persistência para 2025: Norte 0,72, Sul 0,69, Nordeste 0,61, Sudeste 0,43, Centro-Oeste 0,36. O Sul volta a aparecer como risco alto apesar de taxa relativamente alta.

Ranking municipal em `reports/risco_metas_futuras_municipios.csv`.

## Enriquecimento usado na modelagem

Fatores e SHAP usam PIB, população, Censo Escolar e Atlas. Consultas em `src/ingestao/consultas_enriquecimento.sql`.
