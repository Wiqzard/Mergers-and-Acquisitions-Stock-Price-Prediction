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

What is Mergers & Acquisitions? 
An acquisition takes place when a company purchases most or all of another companies shares in order to gain control of that company. Usually 50% of a targets share stock are enough to allows the acquirer to gain control about the newly acquired assets without approval of the company's other shareholders.
One speaks of a merger  when a brand new entity is formed from two seperate companies. Therefore the main difference in an acquisition is, that the parent company fully takes over the target company and integrates it into the parent company. While in a merger, the two companies combine and create a brand new entity. 

There are several different types of acquisitions. Three of which are especially important for this project (more on that later):
* Vertical: the parent company acquires a company that is somewhere along its supply chain.
* Horizontal: the parent company buys a competitor or other firm in their own industry sector, and at the same point in the supply chain.
* Congeneric: also known as a market expansion, this occurs when the parent buys a firm that is in the same or a closely-related industry, but which has different business lines or products.

Reasons why a company would want to acquire other companies often are amongst diversification, greater market share, increased synergy, cost reductions, as a way to enter a foreign market, as a growth strategy, to decrease competition or to gain new technology.
However acquistions can also involve rists for the acquirer, most common one are integration risk, overpayment and culture clash. 

Before and after an acquisiton we can investigate the stock prices of the involved companies.
In the case of the target company often shares will rise to a level close to that of the acquirer’s offer, assuming of course that the offer represents a significant premium to the target’s previous stock price. In fact, the target’s shares may trade above the offer price if the perception is either that the acquirer has low-balled the offer for the target and may be forced to raise it, or that the target company is coveted enough to attract a rival bid.
However the price can also happen to be traded below the announced offer price. This generally occurs when part of the purchase in made in the acquirer's shares and the stock plummets when the deal is allowed. Generally with information about the deal the resulting stock price of the target company can be estimated.
For the acquirer, the impact of an M&A transaction depends on the deal size relative to the company’s size. Such risks could be integration risk
overpayment, cilture clash. The larger the potential target, the bigger the risk to the acquirer. A company may be able to withstand the failure of a small-sized acquisition, but the failure of a huge purchase may severely impede its long-term success.

Therefore it is not far fetched to assume that the stock price of a company before and after the acquisition took place exhibits certain trends that can be possibly predicted and observed. Based on the above backrground of M&A, I assume that for the scope of this project and the for me accesible data, the most important measurable indices, that have the most predictive power for stock price movements,  are share values in a certain time span before the acquisition takes place, and the value of the acquirer and target company.





## Methods
 - Data Sources
    My research yield that there are mainly 3 data sources one could utilize to get the data mentioned above.
    - The easiest way would be paid services like Bloomberg, S&P Capital IQ, Refinitiv Workspace, SDC Platinum. However very costly.
    - SEC Filings M&A must be anounced 4 days berfore due date. And succesfull M&A within 71 days after. in 8-K filing. Also in annual report 10-k under         Part II, Item 8:Financial Statements and Supplementary Data, Note 8: Acquisitions. As an example: https://www.sec.gov/ix?doc=/Archives/edgar/data/0000789019/000156459022026876/msft-10k_20220630.htm.
    The Problem with this approach lies in the fact that many companies have different standards in filling out the files. This means that the data is not consistently structured, characters. This makes the data parsing either impossible with conventional web scraping techniques or to a NLP problem (future project).
    - Kaggle MAA Dataset https://www.kaggle.com/datasets/shivamb/company-acquisitions-7-top-companies
     The data set contains following information: Parent Company/ Acquisition Year/Month / Acquired Company/ Business/ Country/ Acquisition/ Price/ Category /Derived Products 
     for Microsoft, Google, IBM, Hp, Apple, Amazon, Facebook, Twitter, eBay, Adobe, Citrix, Redhat, Blackberry and Disney.
    - The individual stock data 150 days prior and 150 days adter the acquisition took place is requested from the yfinance API.
  
 - Data Wrangling/ Database
    - The Kaggle data only provided acquisition year and month. Thus took as event date middle of month, considering resulting bias/variance.
    - Some duplicates.
    - The stock data was partially incomplete. Some of the in total 300 character long sequences of stock data were completely empty, or partially empty for instance every weekend. If the intervals of missing data was not longer than 4, they got filled with the pre and perceeding data equally. The other ones got discarded.
    - Only roughly 10% of the acquisition events provided information about the acquisition price.
    
    - The data has been uploaded to a database hosted on Amazon AWS.
    - The SQL and data parsing was exercised with the python framework pymysql
    - 
 - Transformer / Informer Network
   - Why transformer  . attention... why positional encoding, time embedding... embedding space etc., feature imput.
   - Testing of Transformer
 - Training  different modes (classiscal, sliding window, autoregressive as in informer)
 - training either autoregressive informer in one forward passlike or standard sliding window and predicting n steps a head show sources why ..
 - Inference: SOS token replaced with label_len last encoder input.(only explain the one)

## Results
  - Show good results short and long term.
  - > Some good some bad, maybe make something like correalation or smth to show if it is usable for arbitrage or not.
## Prospects
  - What can be done better?
  - Mainly more data, more KPI's like... Temporal Fusion Transformer
## Sources
