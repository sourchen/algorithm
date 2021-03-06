<div align="center" id="login-optional-fields">
	<img src="https://github.com/sourchen/algorithm/blob/master/media/CTDSSicon.png" alt="Logo" width="250" height="250">
	<h3 align="center">Carry on For You / Carry Trade Decision Support System (CTDSS)</h3>
	<p align="center">&#128176; prediction short- to mid-term exchange rates</p>
	<p align="center">&#9935; self-developed learning ANN algorithm</p>
	<p align="center">&#128521; User friendly interface design</p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
	  <li>
		  <a href="#about-the-project">About CTDSS</a>
		  <ul>
				<li><a href="#Introduction">Introduction</a></li>
				<li><a href="#System_Functions">System Functions</a></li>
				<li><a href="#built-with">Built With</a></li>
	    </ul>
    </li>
	  <li>
		<a href="#Model Development Techniques">Model Development Techniques</a>
		  <ul>
				<li><a href="#Data Description">Data Description</a></li>
				<li><a href="#Model Description">Model Description</a></li>
		  </ul>
	</ol>
</details>

<h2 id="about-the-project">About CTDSS</h2>

<h3 id="Introduction">&#128205; Introduction</h3>

***Carry Trade***

Carry trade is involved a high-yielding country and a low-yielding one, while traders attempt to capture the difference between their rates. It has become a source of investment strategy for investors or those with a natural need for forex trading. <b><u>Profits made from a carry trade</u></b>derive from not only the difference between interest rates but also <b><u>the increase or decrease of exchange rate due to forward rate bias</u></b>. 

***Problem Solving***

Targeting companies that need foreign exchange hedge and capital circulation, we position our system as a carry trade decision support system (CTDSS). By <b><u>building prediction models of multiple currency pairs</u></b>and <b><u>returning with short- to mid-term exchange rates</u></b>, we provide users with ancillary values to proceed with a carry trade.

<p align="right">(<a href="#login-optional-fields">back to top</a>)</p>

---

<h3 id="System_Functions">&#128205; System Functions</h3>

<h4>&#128161; Key information imparted for quick data query</h4>
	
CTDSS provides five currencies (Table 1) and three transaction durations (3, 6, and 12 months), forming 75 (= 5*5*3) forex trade combinations.
	
As shown in Figure 1, relevant indicators are well presented in a table format in which we collect the data from major banks of their countries and store it in our database.

<div align="center">
<table>
	<tr><th>Currencies</th><th>Representative banks</th></tr>
	<tr> <td>Euro (EUR)</td><td>European Central Bank</td></tr>
	<tr> <td>Indonesian Rupiah (IDR)</td><td>Bank Mandiri</td></tr>
	<tr> <td>Japanese Yen (JPY)</td><td>MUFG Bank</td></tr>
	<tr> <td>New Taiwan Dollar (TWD)</td><td>Bank of Taiwan</td></tr>
	<tr> <td>United States Dollar (USD)</td><td>JPMorgan Chase</td></tr>
</table>
<p>Table 1. Five Currencies and Corresponding Representative Banks</p>
</div>

<p align="center">
  <img src="https://github.com/sourchen/algorithm/blob/master/media/p1.gif?raw=true" alt="Sublime's custom image" width="600"/>
</p>
<p align="center">Figure 1. System Homepage of CTDSS</p>

<h4>&#128200; Historical Data Visualization and Exchange Rate Calculator</h4>

 As shown in Figure 2.
1. Trends of historical data including loan rate, deposit rate, and spot exchange rate are shown in the line chart, so are our predicted exchange rate results.
2. An assistive exchange rate calculator can convert between the currency calculated by either the current spot exchange rate or the exchange rate predicted by our model.
	
<p align="center">
  <img src="https://github.com/sourchen/algorithm/blob/master/media/p2.gif?raw=true" alt="Sublime's custom image" width="600"/>
</p>
<p align="center">Figure 2. Historical Data Chart (Left); Exchange Rate Conversion (Right)</p>

<h4>&#128185; Capital Circulation in carrying Trade Transaction</h4>

Since several numerical values are included in carrying trade, we have come up with the interface shown in Figure 3, offering users to input the amount and returning a comprehensible outline of the transaction. 


<p align="center">
  <img src="https://github.com/sourchen/algorithm/blob/master/media/p3.gif?raw=true" alt="Sublime's custom image" width="600"/>
</p>
<p align="center">Figure 3. Capital Circulation in carrying Trade Transaction</p>

<p align="right">(<a href="#login-optional-fields">back to top</a>)</p>

---

<h3 id="built-with">&#128205; Built With</h3> 

Built CTDSS system: use Django as system framework along with Python as back-end.
Implement a new learning algorithm: use TensorFlow (version 2.5) and Python.


* [Python](https://www.python.org)
* [Django](https://www.djangoproject.com)
* [HTML5](https://html5.org)
* [CSS](https://www.w3.org)
* [JavaScript](https://www.javascript.com)
* [TensorFlow](https://www.tensorflow.org)

<p align="right">(<a href="#login-optional-fields">back to top</a>)</p>

---

<h2 id="Model Development Techniques">Model Development Techniques</h2>

<h3 id="Data Description">&#128205; Data Description</h3> 

- **variables**
The independent variables for predicting forward exchange rate are based on macroeconomics theory (Jorda, 2012), Table 2 are 4 input variables: 

<div align="center">
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
</div>

- monthly data: January 2000 - August 2021 (248 months)
	- training data: January 2000 ??? March 2017 (196 months)
	- testing set: April  2017 ??? August 2022 (52 months)

- **data preprocessing**
	* log: ensure all variables are stationary time-series
	* standardization

<p align="right">(<a href="#login-optional-fields">back to top</a>)</p>

---

<h3 id="Model Description">&#128205; Model Description</h3> 

The low chart of our self-developed learning algorithm for the exchange rate prediction model is shown in Figure 4.

***How this algorithm differs from other neural network models?***
>The difference is the way they learn data.
>
In general, the hidden layer nodes of the neural network model are set directly at the beginning. However, Our self-developed learning algorithm can adaptly learn based on current performance it learned to decide whether add or delete nodes. As for some data which the model can't learn right away, it'll use a rule-based method to cram and adjust hyperparameters of the node. Adapt learning can avoid the overfitting condition often encountered in neural network learning.

<p align="center">
  <img src="https://github.com/sourchen/algorithm/blob/master/media/p4.png?raw=true" alt="Sublime's custom image" width="600"/>
</p>
<p align="center">Figure 4. Flow chart of our self-developed learning algorithm</p>

***How well does the model train? Explain why.***
>The result is shown in Figure 5. This algorithm has better performance in TWD/USD, TWD/IDR, and the research results show that they are all better than general statistical models (ex. multiple regression).

<div align="center">
<table>
	<tr><th>currency pair / period</th><th>3 months</th><th>6 months</th><th>1 year</th></tr>
	<tr> <td>USD/TWD</td><td>1.01%</td><td>1.074%</td><td>1.379%</td></tr>
	<tr> <td>USD/EUR</td><td>2.214%</td><td>4.978%</td>
	<td>9.642%</td></tr>
	<tr> <td>USD/IDR</td><td>2.34%</td><td>2.754%</td><td>3.268%</td></tr>
	<tr> <td>USD/JPY</td><td>1.208%</td><td>1.817%</td><td>1.871%</td></tr>
	<tr> <td>EUR/TWD</td><td>1.959%</td><td>3.354%</td><td>5.891%</td></tr>
	<tr> <td>EUR/IDR</td><td>3.516%</td><td>3.365%</td><td>4.352%</td></tr>
	<tr> <td>EUR/JPY</td><td>3.64%</td><td>3.649%</td><td>3.687%</td></tr>
	<tr> <td>TWD/JPY</td><td>2.015%</td><td>2.163%</td><td>2.498%</td></tr>
	<tr> <td>TWD/IDR</td><td>2.79%</td><td>2.239%</td><td>3.733%</td></tr>
	<tr> <td>JPY/IDR</td><td>4.387%</td><td>3.704%</td><td>3.154%</td></tr>
</table>
<p>Table 3. our self-developed learning algorithm performance???sMape???</p>
</div>

After our study, it was found that there is higher interdependence with the currency markets of Taiwan, the United States, and Indonesia. Therefore, the macroeconomic data we used can comprehensively present the current currency market.

On the contrary, due to the low-interest-rate policy of Europe and Japan, training data cannot reflect overall currency markets. Hence, in the future, we would like to have a deeper study on whether other political or societal variables are suitable for the prediction of these currency markets influenced by policies.

<p align="center">
  <img src="https://github.com/sourchen/algorithm/blob/master/media/evaluation.png?raw=true" alt="Sublime's custom image" width="600"/>
</p>
<p align="center">Table 4. model performance comparsion???sMape???</p>

<p align="right">(<a href="#login-optional-fields">back to top</a>)</p>
