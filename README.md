# Neural Network
從James D. McCaffrey在微軟部落格上的C#開放原始碼修改而來。

我的碩士論文原本是使用這個程式碼的C#版本，但是進公司後必須使用Linux，為了能夠使用以前的研究成果，因此必須修改成C++版本。

main.cpp 裏面是一個測試範例，對sin(x)在0~2pi之間隨機取樣，讓類神經網路從這些樣本去學習，嘗試訓練成一個能夠在[0,2pi]區間準確regression出sin(X)的神經網路。

本類別亦可修改為Classifier model，只要把output layer加上活化函數sigmoid，輸出數值即可滿足兩大機率公設。 
