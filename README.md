# Neural Network

<b>沿革與動機</b><br>
這個類別從James D. McCaffrey在微軟部落格上的C#開放原始碼修改而來。<br>
我的碩士論文原本是使用這個程式碼的C#版本，但是進公司後必須使用Linux，為了能夠沿用過去的研究成果，因此必須將NeuralNetwork類別修改成C++版本。<br>
<br>
<b>Demo說明</b><br>
main.cpp 裏面是一個測試範例，對sin(x)在0~2pi之間隨機取樣，讓類神經網路從這些樣本去學習，嘗試訓練成一個能夠在[0,2pi]區間準確regression出sin(X)的神經網路。<br>
<br>
<b>NN架構說明</b><br>
Input Layer: 輸入節點數可以任意設定。<br>
Hidden Layer: 目前限定一層，寬度可以任意設定。<br>
Output Layer: 輸出節點數可以任意設定，此層單純加總，無活化函數<br>
倒傳遞: 採用梯度下降法。<br>
<br>
<b>如何修改為Classifier Model？</b><br>
本類別為Regression Model，亦可修改為Classifier Model。<br>
只要把output layer加上活化函數softmax，輸出數值即可滿足下述兩大機率公設。<br>
*0 <= P <= 1
*Σ(P) == 1
