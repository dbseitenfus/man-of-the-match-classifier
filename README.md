# Man of the Match Classifier

Projeto de estudos de Machine Learning para prever o **Jogador da Partida** (Man of the Match) da Copa do Mundo FIFA 2018.

## Objetivo

Classificar, com base em estatísticas das partidas da FIFA 2018, qual time ou jogador tem o perfil de "Man of the Match" — utilizando técnicas de ML interpretável para entender **quais features mais influenciam** a predição.

## Dataset

`FIFA 2018 Statistics.csv` — contém estatísticas numéricas das partidas da Copa do Mundo FIFA 2018.

- **Target:** coluna `Man of the Match` (convertida de `"Yes"`/`"No"` para binário)
- **Features:** todas as colunas de tipo numérico

## Modelos

| Modelo | Biblioteca |
| --- | --- |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` |

## Interpretabilidade

### Permutation Importance (eli5)

Mede o quanto a performance do modelo cai ao embaralhar cada feature — identifica as variáveis mais relevantes.

### Partial Dependence Plots (pdpbox)

Visualiza como a variável `Distance Covered (Kms)` impacta a probabilidade de ser Man of the Match, isolando o efeito de cada feature.

## Estrutura

```text
.
├── main.py                    # Script principal
├── FIFA 2018 Statistics.csv   # Dataset
└── README.md
```

## Dependências

```bash
pip install numpy pandas scikit-learn eli5 pdpbox matplotlib
```

No macOS, instale também o OpenMP (necessário para o xgboost, dependência do eli5):

```bash
brew install libomp
```

## Como Executar

```bash
python main.py
```

O script irá:

- Carregar e preparar os dados
- Treinar Decision Tree e Random Forest
- Avaliar com matriz de confusão e acurácia
- Exibir Permutation Importance para ambos os modelos
- Plotar PDPs para `Distance Covered (Kms)`
