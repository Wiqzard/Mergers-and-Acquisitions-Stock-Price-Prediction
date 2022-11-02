# Mergers & Acquisitions Stock Price Prediction
[![forthebadge](https://forthebadge.com/images/badges/built-with-science.svg)](https://forthebadge.com)
[![Gem Version](https://badge.fury.io/rb/colorls.svg)](https://badge.fury.io/rb/colorls)
## Table of contents
- [Economic Background & Motivation](#motivation)
- [Methods](#methods)
- [Results](#results)
- [Prospects](#prospects)
- [Sources](#sources)

## Economic Background & Motivation

What are Mergers & Acquisitions? 

An acquisition takes place when a company purchases most or all of another company's shares to gain control of that company. Usually, 50% of a target's share stock is enough to allow the acquirer to gain control of the newly acquired assets without the approval of the company's other shareholders.

One speaks of a merger when a new entity is formed from two separate companies. Therefore the main difference in an acquisition is, that the parent company fully takes over the target company and integrates it into the parent company. In a merger, two companies merge and become one. 

There are several different types of acquisitions. Three of which are especially relevant for this project:

* Vertical: the parent company acquires a company that is somewhere along its supply chain.

* Horizontal: the parent company buys a competitor or other firm in their industry sector, and at the same point in the supply chain.

* Congeneric: also known as a market expansion, this occurs when the parent buys a firm that is in the same or a closely-related industry, but which has different business lines or products.

A company may be interested in acquiring other firms for various reasons, including diversification, additional market share, increased synergy, cost reductions, entry into a foreign market, a growth strategy, to decrease competition, or to acquire new technology.

However acquisitions can also involve risks for the acquirer. The most common issues are integration risk, overpayment, and culture clash. 

Before and after an acquisition we can investigate the stock prices of the involved companies.

In the case of the target company, shares often will rise to a level close to that of the acquirer’s offer. This assumes of course that the offer represents a significant premium to the target’s previous stock price. In fact, the target’s shares may trade above the offer price if the perception is either that the acquirer has low-balled the offer for the target and may be forced to raise it, or that the target company is coveted enough to attract a rival bid.

Prices can also be traded below the announced offer price. This generally occurs when part of the purchase is made in the acquirer's shares and the stock plummets when the deal is allowed. Generally, with information about the deal, the resulting stock price of the target company can be estimated.

For the acquirer, the impact of an M&A transaction depends on the deal size relative to the company’s size. The larger the potential target, the bigger the risk to the acquirer. A company may be able to withstand the failure of a small-sized acquisition, but the failure of a huge purchase may severely impede its long-term success.

Therefore it is not far-fetched to assume that the stock price of a company before and after the acquisition took place exhibits certain trends that can be possibly predicted and observedFrom my understanding of the above background of M&As, I assume, for the scope of the study and the available data for me, that the most significant measurable indices to forecast stock price movements are share prices in a certain period before the acquisition and the values of the acquirer and target company.

## Methods

 - Data Sources

    There are mainly 3 data sources one could utilize to get the data mentioned above.

    - The easiest way would be paid services like Bloomberg, S&P Capital IQ, Refinitiv Workspace, and SDC Platinum.

    - Next, there is official government data. SEC Filings M&A must be announced 4 days before the acquisition date. And successful M&A deals within 71 days after in an 8-K filing or annual report 10-K filing. As an example see: https://www.sec.gov/ix?doc=/Archives/edgar/data/0000789019/000156459022026876/msft-10k_20220630.htm.

    The Problem with this approach lies in the fact that many companies have different standards in filling out the files. It means that the data is not consistently structured, so parsing it either becomes impossible with conventional web scraping techniques or becomes a problem for NLP tools.

    - Lastly: Kaggle MAA Dataset https://www.kaggle.com/datasets/shivamb/company-acquisitions-7-top-companies

     The data set contains following information: Parent Company/ Acquisition Year/Month / Acquired Company/ Business/ Country/ Acquisition/ Price/ Category /Derived Products 

     for Microsoft, Google, IBM, Hp, Apple, Amazon, Facebook, Twitter, eBay, Adobe, Citrix, Redhat, Blackberry, and Disney.

    - The individual stock data 150 days prior and 150 days after the acquisition took place is requested from the yfinance API.

  

 - Data Wrangling/ Database

    - The Kaggle data only provided acquisition year and month. Thus I selected as the acquisition date the middle of the month, considering the resulting bias/variance.

    - The stock data was partially incomplete. Some of the total 300-character long sequences of stock data were completely empty, or partially empty. If the intervals of missing data were not longer than 4, they got filled with the pre and preceding data equally. The other ones got discarded along with the duplicate sequences.

    - Only roughly 10% of the acquisition events provided information about the acquisition price.

    - The cleaned data has been uploaded to a relational database hosted on Amazon AWS RDS.

    - The SQL and data parsing was exercised with the python framework pymysql

 - Transformer / Informer Network

    - Because of their attention mechanism Transformer networks have been proven in many recent publications to be efficient in capturing long-term   

   processes in time series forecasting. To capture positional and temporal data, positional encoding from "Attention is all you need" and time2vec time 

   embedding is implemented.

   - 2 different modes have been tested. One where the transformer predicts the n days starting from the end of the input sequence in an autoregressive 

  manner, i.e. in one forward pass. And classically, where the next time point following the input sequence is predicted which gets fed into the 

  transformer for predicting multiple steps.

   - In the former during inference, the SOS token gets replaced with the label_len last encoder inputs.
## Results(Probably still duplicates in data blurring results)
* Best results achieved with InformerStack(enc_in=1, dec_in=1, c_out=1, seq_len=seq_len, label_len=label_len, out_len=pred_len, 
                 factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=2048, 
                 dropout=0) on 120 epochs:
  - On the training set:
     - avg_R2: 0.901  ---  avg_MAE: 0.243   ---   avg_MSE: 0.189 (unscaled)
  - On the test set:
     - avg_R2: 0.034  ---  avg_MAE: 1.59   ---   avg_MSE: 9.44   (unscaled)
     - avg_R2: 0.034  ---  avg_MAE: 0.439   ---   avg_MSE: 0.357 (standard scaled)
     
  *Good result on the test set*
  ![alt text](https://github.com/Wiqzard/Mergers-and-Acquisitions-Stock-Price-Prediction/blob/master/pics/ok2.png)
  *Bad result on the test set*
  ![alt text](https://github.com/Wiqzard/Mergers-and-Acquisitions-Stock-Price-Prediction/blob/master/pics/bad.png)
  *Statistics of data*
  ![alt text](https://github.com/Wiqzard/Mergers-and-Acquisitions-Stock-Price-Prediction/blob/master/pics/statistics1.png)
  ![alt text](https://github.com/Wiqzard/Mergers-and-Acquisitions-Stock-Price-Prediction/blob/master/pics/statistics2.png)
  
  - Comparison with ARIMA model fitted to individual M&A's:
     - avg_MAE: 2.87  ---   avg_MSE: 50.1
## Prospects
- As expected results show only granular trends for a medium ranged prediction (around 50 days)
  - What can be done better?
  - The main problem was the lack of data.
  - Gather more data, especially with features such as acquiree value, acquirer value, deal price, etc.
  - Use the data as an additional input to the output layer.
  - Use random window of stock date where no M&A occured to gauge the effect.
## Sources

* https://arxiv.org/pdf/2205.01138.pdf
* https://arxiv.org/pdf/2001.08317.pdf
* https://arxiv.org/pdf/2012.07436.pdf
* https://arxiv.org/pdf/2204.11115.pdf
* https://arxiv.org/pdf/2012.07436.pdf
* https://github.com/zhouhaoyi/Informer2020
