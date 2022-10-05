# Mergers & Acquisitions Stock Price Prediction(IN PROGRESS)
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
One speaks of a merger when a new entity is formed from two separate companies. Therefore the main difference in an acquisition is, that the parent company fully takes over the target company and integrates it into the parent company. While in a merger, the two companies combine and create a brand new entity. 

There are several different types of acquisitions. Three of which are especially important for this project:
* Vertical: the parent company acquires a company that is somewhere along its supply chain.
* Horizontal: the parent company buys a competitor or other firm in their industry sector, and at the same point in the supply chain.
* Congeneric: also known as a market expansion, this occurs when the parent buys a firm that is in the same or a closely-related industry, but which has different business lines or products.

Reasons, why a company would want to acquire other companies often, are diversification, extended market share, increased synergy, cost reductions, a way to enter a foreign market, as a growth strategy, to decrease competition, or to gain new technology.
However acquisitions can also involve risks for the acquirer, the most common ones are integration risk, overpayment, and culture clash. 

Before and after an acquisition we can investigate the stock prices of the involved companies.
In the case of the target company, shares often will rise to a level close to that of the acquirer’s offer, assuming of course that the offer represents a significant premium to the target’s previous stock price. In fact, the target’s shares may trade above the offer price if the perception is either that the acquirer has low-balled the offer for the target and may be forced to raise it, or that the target company is coveted enough to attract a rival bid.
However, the price can also happen to be traded below the announced offer price. This generally occurs when part of the purchase in made in the acquirer's shares and the stock plummets when the deal is allowed. Generally, with information about the deal, the resulting stock price of the target company can be estimated.
For the acquirer, the impact of an M&A transaction depends on the deal size relative to the company’s size. The larger the potential target, the bigger the risk to the acquirer. A company may be able to withstand the failure of a small-sized acquisition, but the failure of a huge purchase may severely impede its long-term success.

Therefore it is not far-fetched to assume that the stock price of a company before and after the acquisition took place exhibits certain trends that can be possibly predicted and observed. Based on the above background of M&As, I assume that for the scope of this project and the for me accessible data, the most important measurable indices, that have the most predictive power for stock price movements,  are share values in a certain period before the acquisition takes place and the values of the acquirer and target company.





## Methods
 - Data Sources
    There are mainly 3 data sources one could utilize to get the data mentioned above.
    - The easiest way would be paid services like Bloomberg, S&P Capital IQ, Refinitiv Workspace, and SDC Platinum.
    - Next, there is official government data. SEC Filings M&A must be announced 4 days before the acquisition date. And successful M&A deals within 71 days after in an 8-K filing or annual report 10-k filing. As an example see: https://www.sec.gov/ix?doc=/Archives/edgar/data/0000789019/000156459022026876/msft-10k_20220630.htm.
    The Problem with this approach lies in the fact that many companies have different standards in filling out the files. This means that the data is not consistently structured, which makes the data parsing either impossible with conventional web scraping techniques or to a NLP problem.
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

## Prospects
- As expected results show only granular trends for a medium ranged prediction (around 50 days)
  - What can be done better?
  - The main problem was the lack of data.
  - Build the model upon architectures like Temporal Fusion Transformer which have been developed specifically for time series tasks.
## Sources

* https://arxiv.org/pdf/2205.01138.pdf
* https://arxiv.org/pdf/2001.08317.pdf
* https://arxiv.org/pdf/2012.07436.pdf
* https://arxiv.org/pdf/2204.11115.pdf
