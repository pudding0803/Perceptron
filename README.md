# Perceptron

> 類神經網路的 project 1

* GUI 介面預覽
![](https://i.imgur.com/82Hl6Lu.png)

* 具有任意維度二元線性分割的功能
    * 二、三維資料具有視覺化的功能
    * 二維資料另外會有每 10% 的訓練結果

* 可隨意建構隱藏層層數、任一隱藏層神經元數
    * 目前傳遞使用的函數只有線性分割，因此使用只會導致結果不準確
    * 目前輸出層固定為一個神經元

* 神經網路設計
    * Activation function 使用 Sigmoid 函數，以 0.5 為界線進行二元分割
    * 符號： $j$ 為當層神經元， $i$ 為前一層神經元， $k$ 為後一層神經元
    * Forward propagation
        * $v_j = \sum(w_{ji} \cdot y_i)$
        * $y_j = \frac{1}{1 + e^{-v_j}}$
    * Backward propagation
        * 若神經元 j 在輸出層： $\delta_j = (d_j - O_j) \cdot O_j \cdot (1 - O_j)$
        * 若神經元 j 在隱藏層： $\delta_j = y_j \cdot (1 - y_j) \cdot \sum(\delta_k \cdot w_{kj})$
        * $w_{ji} \leftarrow w_{ji} + \eta \cdot \delta_j \cdot y_i$

* 主要使用的函式庫：numpy、plotly、PyQt5

* 輸入資料的格式範例：
    * 一行為一筆資料，範例為 $m$ 筆 $n$ 維的資料， $d$ 為期望輸出

$$
  \begin{matrix}
   x_{11} & x_{12} & \cdots & x_{1n} & d_1 \\
   x_{21} & x_{22} & \cdots & x_{2n} & d_2 \\
   \vdots & \vdots & \vdots & \vdots & \vdots\\
   x_{m1} & x_{m2} & \cdots & x_{mn} & d_m \\
  \end{matrix}
$$

* 問題
    * 不支援超過兩群的分類
    * 不支援非線性可分割
    * 執行檔視窗的 icon 似乎會消失
