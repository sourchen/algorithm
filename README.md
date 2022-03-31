<div align="center">
    <img src="https://github.com/sourchen/algorithm/blob/master/CTDSSicon.png" alt="Logo" width="200" height="200">
  <h3 align="center">Carry On For You / Carry trade support sysytem</h3>
  <p align="center">Based on prediction of short- to mid-term exchange rates, with self develop ANN algorithm</p></div>

## About CTDSS

Carry trade is involved a high-yielding country and a low-yielding one, while traders attempt to capture the difference between their rates. It has become a source of investment strategy for investors or those with a natural need for forex trading. Profits made from a carry trade derive from not only the difference between interest rates but also the increase or decrease of exchange rate due to forward rate bias. 

Targeting companies that need foreign exchange hedge and capital circulation, we position our system as a carry trade decision support system (CTDSS). <abbr title="Hyper Text Markup Language"><font color=blue>**By building prediction models of multiple currency pairs and returning with short- to mid-term exchange rates,**</font> we provide users with ancillary values to proceed with a carry trade.

### System Introduction

Here are the system functions:

* **Key information imparted for quick data query**
CTDSS provides five currencies (Table 1) and three transaction durations (3, 6, and 12 months), forming 75 (= 5*5*3) forex trade combinations. 
	
![image](https://github.com/sourchen/algorithm/blob/master/media/p1.gif)
	
	As shown in Figure 1, relevant indicators are well presented in a table format in which we collect the data from major banks of their countries and store it in our database.

| Currencies | Representative banks  |
| :----   | :-----  |
|  Euro (EUR)      | European Central Bank |
| Indonesian Rupiah (IDR) |   Bank Mandiri   |
| Japanese Yen (JPY) |    MUFG Bank    |
| New Taiwan Dollar (TWD)  |    Bank of Taiwan    |
| United States Dollar (USD) |    JPMorgan Chase    |

<p align="center">Table 1. Five Currencies and Corresponding Representative Banks</p>

* **Historical Data Visualization and Exchange Rate Calculator**
 As shown in Figure 2.
	* Trends of historical data including loan rate, deposit rate, and spot exchange rate are shown in the line chart, so are our predicted exchange rate results.
	* An assistive exchange rate calculator can convert between the currency calculated by either the current spot exchange rate or the exchange rate predicted by our model.

* **Capital Circulation in Carry Trade Transaction**
Since several numerical values are included in a carry trade, we have come up with the interface shown in Figure 3, offering users to input the amount and returning a comprehensible outline of the transaction. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

Built CTDSS system: use Django as system framework along with Python as back-end.
Implement a new learning algorithm: use TensorFlow (version 2.5) and Python.


* [Python](https://www.python.org)
* [Django](https://www.djangoproject.com)
* [HTML5](https://html5.org)
* [CSS](https://www.w3.org)
* [JavaScript](https://www.javascript.com)
* [TensorFlow](https://www.tensorflow.org)

<p align="right">(<a href="#top">back to top</a>)</p>

## Model Development Techniques

### Data Description

- **variables**
The independent variables for predicting forward exchange rate are based on macroeconomics theory (Jorda, 2012), Table 2 are 4 input variables: 

<table>
	<tr><th>Variable</th><th>Definition</th></tr>
	<tr> <td>x1</td><td>logged exchange rate (home currency / foreign currency)</td></tr>
	<tr> <td>x2</td><td>nominal interest rate abroad - nominal interest rate home</td></tr>
	<tr> <td>x3</td><td>real exchange rate abroad - real exchange rate home</td></tr>
	<tr> <td>x4</td><td>domestic inflation rate abroad - domestic inflation rate home</td></tr>
	<tr><th colspan="3">Output</th></tr>
	<tr> <td>y</td><td>the prediction for exchange rate</td></tr>
</table>
<p align="center">Table 2. Variables based on macroeconomics theory</p>

- monthly data: January 2000 - August 2021 (248 months)
	- training data: January 2000 – March 2017 (196 months)
	- testing set: April  2017 – August 2022 (52 months)

- **data preprocessing**
	* log: ensure all variables are stationary time-series
	* standardization
	
### Model Description
The low chart of our self-developed learning algorithm for the exchange rate prediction model is shown in Figure 4.

***How this algorithm differs from other neural network models?***
>The difference is the way they learn data.
>
In general, the hidden layer nodes of the neural network model are set directly at the beginning. However, Our self-developed learning algorithm can adaptly learn based on current performance it learned to decide whether add or delete nodes. As for some data which the model can't learn right away, it'll use a rule-based method to cram and adjust hyperparameters of the node. Adapt learning can avoid the overfitting condition often encountered in neural network learning.

<p align="center">Figure 4. Flow chart of our self-developed learning algorithm</p>

- **Model Evaluation**

***How well does the model train? Explain why.***
>This algorithm has better performance in TWD/USD, TWD/IDR, and the research results show that they are all better than general statistical models (ex. multiple regression).

After our study, it was found that there is higher interdependence with the currency markets of Taiwan, the United States, and Indonesia. Therefore, the macroeconomic data we used can comprehensively present the current currency market.

On the contrary, due to the low-interest-rate policy of Europe and Japan, training data cannot reflect overall currency markets. Hence, in the future, we would like to have a deeper study on whether other political or societal variables are suitable for the prediction of these currency markets influenced by policies.
