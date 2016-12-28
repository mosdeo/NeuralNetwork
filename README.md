# Neural Network

<b>沿革與動機</b><br>
這個類別從James D. McCaffrey在微軟部落格上的C#開放原始碼修改而來的C++版本。<br>
我的碩士論文原本是使用這個程式碼的C#版本。進公司後為了能夠將過去的研究成果沿用在Linux上，因此必須將NeuralNetwork類別修改成C++版本。<br>
<br>
<b>Demo說明</b><br>
main.cpp 裏面是一個測試範例，對sin(x)在0~2pi之間隨機取樣，讓類神經網路從這些樣本去學習，嘗試訓練成一個能夠在[0,2pi]區間準確regression出sin(X)的神經網路。<br>
<br>
<b>NN架構說明</b><br>
Input Layer: 輸入節點數可以任意設定。<br>
Hidden Layer: 這個 branch 是2隱藏層，目前還是實驗性的，寬度可以任意設定，但是兩層必須一樣寬。<br>
Output Layer: 輸出節點數可以任意設定。<br>

<br>
<b>如何修改為Classifier Model？</b><br>
本類別預設為Regression Model，亦可修改為Classifier Model。<br>
只要把output layer加上活化函數Softmax，輸出數值即可滿足下述兩大機率公設。<br>
Softmax在本類別的成員函式有，直接將輸出傳入即可。<br>
<ul>
    <li>0 <= P <= 1</li>
    <li>Σ(P) == 1</li>
</ul>
