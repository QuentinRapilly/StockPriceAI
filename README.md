# StockPriceAI
We aim at predicting stock prices thanks to AI methods and more particularly reinforcement learning.

## RNN approach

The code is located in the directory **RNN_LSTM/**

- **stock_price_prediction_RNN.ipynb**: contains the notebook that allows the LTSM to learn on S&P500 data, with 2 differents data preprocessing approaches. Everything is described in the notebook : one just have to run the cells and follow experimentations.

## Transformers (ViT) approach

The code is located in the directory **Transformer_ViT**.
- **Config.py**: configuration for network, now coded in the notebook below
- **Stock_ViT.py**: implementation of the network, now coded in the notebook below
- **ViT_dataset.py**: pytorch custom dataset for data preprocesing, now coded in the notebook below
- **ViT_for_stock_pred.ipynb**: contains the notebook that allows the transformer to learn on S&P500 data, in order to compare results with first RNN approach.


## RL approach

The code is located in the directory **RL/**

2 differents approaches are explored:

One first approach with strong hypotheses :
- **Modelisation_RL _2.ipynb**
- **Modelisation_RL.ipynb**
- **Modelisation_RL_3 (1).ipynb**
- **Modelisation_RL_Final.ipynb** : final file where the computation is operational and working.

One refined model with theoretical financial sense :
- **RL_DQNv2.ipynb** : build an environment and an agent capable of trading and choosing the quantity of stocks that it wants to buy or sell, in order to maximize the profits. The state-action function is estimated with a DQN. The modelization and the main results are explained in the notebook.

## Binance API

A binance API pipeline was implemented in order to be able to deal with cryptocurrencies, and test our architectures on them.

## Further works

- Linking the binance API to test our architectures on crypto-currencies

- Refine some hyper-parameters sets, to find the most optimal one, especially on ViT and RLv2

- Comparison of the different models at several level (on S&500 data, on crypto-currencies, on trading simulations, starting with a capital and leading trades)
