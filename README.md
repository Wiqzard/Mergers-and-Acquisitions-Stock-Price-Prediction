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
When acompany purchases most or all of another companies shares in order to gain control of that company. Usually 50% of a targets share stock are enough to allows the acquirer to gain control about the newly acquired assets without approval of the company's other shareholders.
One speaks of a merger  when a brand new entity is formed from two seperate companies. Therefore the main difference in an acquisition is, that the parent company filly takes over the target company and integrates it into the parent company. While in a merger, the two companies combine and create a brand new entity. 

There are several different types of acquisitions. Two of which are especially important in this case since best data(more on that later)
* Vertical: the parent company acquires a company that is somewhere along its supply chain.
* Horizontal: the parent company buys a competitor or other firm in their own industry sector, and at the same point in the supply chain.
* Congeneric: also known as a market expansion, this occurs when the parent buys a firm that is in the same or a closely-related industry, but which has different business lines or products.

Reasons why a company would want to acquire other companies often are economies of scale, diversification, greater market share, increased synergy, cost reductions, As a Way to Enter a Foreign Market, As a Growth Strategy, To Reduce Excess Capacity and Decrease Competition
To Gain New Technology.
However acquistions can involve rists for the acquirer, often such as Integration Risk, Overpayment, Culture Clash. 

Before and after an acquisiton we can investigate the stock prices of the involved companies.
In the case of the target company often shares will rise to a level close to that of the acquirer’s offer, assuming of course that the offer represents a significant premium to the target’s previous stock price. In fact, the target’s shares may trade above the offer price if the perception is either that the acquirer has low-balled the offer for the target and may be forced to raise it, or that the target company is coveted enough to attract a rival bid.
However the price can get lower....
When payed in shares of other company good estimation possible...

For the acquirer, the impact of an M&A transaction depends on the deal size relative to the company’s size. Such risks could be integration risk
overpayment, cilture clash. The larger the potential target, the bigger the risk to the acquirer. A company may be able to withstand the failure of a small-sized acquisition, but the failure of a huge purchase may severely jeopardize its long-term success.

Trefore it is not far fetched to assume that the stock price of a company before and after the acquisition took place exhibits certain trends that can be possibly predicted and observed. Based on the above explanation of M&A I assume that for the scope of this project and the for me accesible data, the most important measurable indices are share values in a certain time span before the acquisition takes place, and the value of the acquirer and target company.









## Methods
 - Data Sources
    - SEC Filings (Talk bs and explain why didnt work)
    - Kaggle MAA Dataset (link)
    - yfinance
    - Dataset: What is conatained? show head or smth.
    - Data Wrangling / cleaning
 - Database
    - Amazon AWS
    - pymysql 
    - 300 - 300 seq -> from which we can form smaller ones if more short or long term stuff favourbalef
 - Transformer / Informer Network
   - Why transformer  . attention... show sources why ..
   - Testing of Transformer
 - Training  different modes (classiscal, sliding window, autoregressive as in informer)
 - Inference  (only explain the one)

## Results
  - Show good results short and long term.
  - > Some good some bad, maybe make something like correalation or smth to show if it is usable for arbitrage or not.
## Prospects
  - What can be done better?
  - Mainly more data, more KPI's like...
## Sources
