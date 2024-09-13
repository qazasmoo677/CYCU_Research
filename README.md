# [論文實做筆記](https://hackmd.io/@YaoT/SyVvwQh6T)
- [Github連結](https://github.com/qazasmoo677/CYCU_Research)
## 檔案路徑參考
> Code
> ├─Model (論文模型以及對照模型的Code)
> &nbsp;│&nbsp;&nbsp;├─GraphRec_pt (論文模型)
> &nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;└─data (實驗比較的資料集)
> &nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ (可以像這樣創不同資料夾存不同參數的final.pkl)
> &nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ v1.1 (use MeanShift 0.001, time 1hr)
> &nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ v1.2 (use MeanShift 0.005, time 1hr)
> &nbsp;│&nbsp;&nbsp;├─RoBERTa_pt (情感語意分析模型)
> &nbsp;│&nbsp;&nbsp;└─對照方法
> &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─1.SVD
> &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─2.CMF
> &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─3.User-based, Item-based
> &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─4.IMP_GCN
> &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─5.DANSER
> &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─6.LightGCN
> └─Preprocessing (資料篩選的Code)
## 前置作業
### CUDA
:::success
#### 說明
- [CUDA 安裝教學](https://medium.com/ching-i/win10-%E5%AE%89%E8%A3%9D-cuda-cudnn-%E6%95%99%E5%AD%B8-c617b3b76deb)
:::
### MongoDB
:::info
#### 說明
- [MongoDB 安裝教學](https://dotblogs.com.tw/explooosion/2018/01/21/040728)
- [MongoDB 語法文件](https://www.mongodb.com/docs/manual/tutorial/query-documents/)
    - 裡面有Python Code可以參考
:::
### 資料篩選
:::warning
#### 相關檔案路徑
> Preprocessing
> ├── DataConverter·py (資料格式轉換 Json -> CSV)
> ├── Preprocessing·ipynb (資料篩選)
> └── Visualization·ipynb (資料視覺化)
#### 資料集
- 資料集使用[Yelp Dataset](https://www.yelp.com/dataset)
#### 軟體版本
- [MongoDB Community Server v7.0.8](https://www.mongodb.com/try/download/community-kubernetes-operator)
- [Navicat Premium 16](https://navicat.com/cht/products/navicat-premium)
#### 環境
- python 3.11.9
#### 套件
- pymongo 4.7.0
- pandas 2.2.2
- numpy 1.26.3
#### 常見問題
1. Yelp Dataset解壓縮之後打不開
    - 解壓縮一次之後在檔名後面加上.tar副檔名解壓縮第二次
2. Yelp Dataset匯不進資料庫
    - 原始檔格式是有點奇怪的JSON，不過pandas可以處理，先轉成csv再匯入，下面有範例
3. 資料表有資料，但query不到
    - 例如要找stars >= 3的review，要先確定review中stars欄位已經轉成int或是decimal，不然預設是string，就query不出來
    - 用下面的資料型態轉換Code跑一下就好了
4. business資料表中的review_count欄位和實際上的review數量不一樣，建議自己跑query查
    - 下面有範例
5. 跑query前要先確定連線跟資料庫是不是正確的
![image](https://hackmd.io/_uploads/rJ4sCWly0.png)
#### Sample Code
- 除了第一個為Python，其他是MongoDB的搜尋語法
##### 1. JSON TO CSV
- 把Yelp從JSON轉成CSV，不然沒辦法匯到MongoDB
```python=
import pandas as pd
import numpy as np
from tqdm import tqdm

# 原始JSON檔名，檔案跟python檔要放同目錄，不然就要改路徑
json_path = 'yelp_academic_dataset_review.json'
# 轉換後的csv檔名
csv_name = "yelp_academic_dataset_review.csv"

df = pd.read_json(json_path, lines=True)
chunks = np.array_split(df.index, 100)
for ix, subset in tqdm(enumerate(chunks)):
    if ix == 0:
        df.loc[subset].to_csv(csv_name, mode='w', index=True)
    else:
        df.loc[subset].to_csv(csv_name, header=None, mode='a', index=True)
```
##### 2. 將business按照州分群，統計數量並遞減排序
```javascript=
db.getCollection("business").aggregate([
    {
        $group: {
            _id: "$state",
            count: { $sum : 1 }
        }
    },
    {
        $sort: {
            count : -1
        }
    }
]);
```

- sort 1是遞增、-1是遞減
##### 3. 資料型態轉換
```javascript=
// 將review資料表中的stars轉成int型態
db.getCollection("review").updateMany(
    {},
    [{
        $set: {
            stars: { $toInt: "$stars" }
        }
    }]
);
```
- 轉成Decimal用$toDecimal
- 轉成Date用$toDate
##### 4. 找在PA州且review數量介於15~200的business
```javascript=
db.getCollection("business").aggregate([
    {
        $match: {
            state: "PA",
        }
    },
    {
        $lookup: {
            from: "review",
            localField: "business_id",
            foreignField: "business_id",
            as: "reviews"
        }
    },
    {
        $addFields: {
            review_count: {
                "$size": "$reviews"
            }
        }
    },
    {
        $match: {
            review_count: {
                $gte: 15,
                $lte: 200
            }
        }
    }
]);
```
- 執行這個後的review_count欄位就會是正確的數量了
- 最後面會有一個新欄位是reviews，裡面會存這個business的所有review，要新增到新表的時候要注意
:::
:::success
#### 資料視覺化 - 原始資料
![image](https://hackmd.io/_uploads/B1Tar1LC6.png)
![image](https://hackmd.io/_uploads/rk_AH1URT.png)
![image](https://hackmd.io/_uploads/rJ1JU1UAT.png)
![image](https://hackmd.io/_uploads/H1N1LJURT.png)
:::
:::danger
#### 我的資料篩選步驟
##### 1. 篩選PA州且在review表中評論數量為15~200的POI到新business表
##### 2. 將第一步的POI對應的評論插入到新review表
##### 3. 刪除新review表中字數<60或評論時間<2017的評論
##### 4. 根據新review表找出評論數15~200的user建立新user表
##### 5. 根據新user表重新篩選review表
##### 6. 根據新review表重新篩選business表
##### * 到後來一定會有一邊不滿足條件(Business或User)
:::
## 模型
### 硬體環境
:::success
- OS: Win11 Professional
- CPU: i9-14900k
- GPU: RTX4090
    - 安裝CUDA時要注意顯卡支不支援
- RAM: 64GB
:::
### RoBERTa情感分析
:::info
#### 說明
[參考資料](https://github.com/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb)
- 先確定有沒有啟用CUDA，速度差很多
#### 相關檔案路徑
> RoBERTa_pt
> └── GetSentimentToMongo.ipynb (評論情感語義分析)
#### 環境
- python 3.11
- CUDA 12.4
#### 套件
- torch
    - 至官網找對應顯卡及cuda版本之pip安裝指令，如下為RTX4090以及cuda12.1對應之指令
    - pip3 install torch torchvision torchaudio \-\-index-url https://download.pytorch.org/whl/cu121
- pandas
- scikit-learn
- seaborn
- transformers
#### 步驟
1. 建立指定版本虛擬環境
    - virtualenv RoBERTa_pt_311 \-\-python=python3.11 
2. 安裝套件
3. 檢查是否有啟用cuda
4. 開始訓練
:::
### 論文模型
:::warning
#### 說明
[參考資料](https://github.com/wenqifan03/GraphRec-WWW19)
#### 相關檔案路徑
> GraphRec_pt
> ├── 0.GraphRec·py (聚合User-POI和社交特徵，沒有聚合POI分群特徵)
> ├── 1.GraphRec_onlyPOI·py (聚合User-POI、社交特徵、POI分群特徵)
> ├── 2.GraphRec_Sentiment·py (聚合User-POI、社交特徵、POI分群特徵、情感語義特徵)
> ├── 3.GraphRec_All·py (聚合User-POI、社交特徵、POI分群特徵、情感語義特徵、時間特徵)
> ├── 98.Preprocessing·ipynb (資料前處理，編碼、刪除空值等等)
> ├── 99.MongoToPKL·ipynb (將MongoDB的資料存成pickle)
> ├── Attention·py (注意力機制)
> ├── Evaluation·py (評估指標)
> ├── POI_Aggregators·py (POI聚合器)
> ├── POI_Encoders·py (POI編碼器)
> ├── SentimentAndTimeGNNCombiner·py (結合原始分數、情感特徵、時間特徵的MLP)
> ├── SentimentGNNCombiner·py (結合原始分數與情感特徵的MLP)
> ├── Social_Aggregators·py (社交關係聚合器)
> ├── Social_Encoders·py (社交關係編碼器)
> ├── TimeGNNCombiner·py (結合原始分數與時間特徵的MLP)
> ├── UV_Aggregators·py (POI-User聚合器)
> ├── UV_Encoders·py (POI-User編碼器)
> └── data
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── （要跑的pickle放這裡）
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── final.pkl
#### 環境
- python 3.6
- CUDA 9.0
#### 套件
- torch
    - 這邊用0.4.1，python=3.6，64位元windows電腦
    - pip3 install https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl
- numpy
- scikit-learn
#### 步驟
1. 建立指定版本虛擬環境
    - 原本那個不知道為啥建不了3.6的，改用這個下面
    - py -3.6 -m venv GraphRec_pt_36
2. 安裝套件
3. 開始訓練
    - 把cmd的目錄指到放code的目錄後執行下面指令
        - python 0.GraphRec.py
    - 後面可以加參數，參數如下
        - \-\-batch_size，訓練的批次，預設為128
        - \-\-embed_dim，嵌入大小，預設為32
        - \-\-lr，學習率，預設為0.001
        - \-\-test_batch_size，測試的批次大小，預設為1000
        - \-\-epochs，要訓練幾個epoch，預設為100
    - 加上參數的指令範例 
        - python 0.GraphRec.py \-\-embed_dim 64 \-\-lr 0.01
:::
### 對照模型1: SVD
:::success
#### 說明
[參考資料](https://github.com/unclebrod/YelpRecommender)
:::
### 對照模型2: PMF
:::info
#### 說明
[參考資料](https://github.com/fuhailin/Probabilistic-Matrix-Factorization/tree/master)
:::
### 對照模型3: User-based, Item-based
:::warning
#### 說明
[參考資料](https://github.com/rahmanidashti/LRSbasics/blob/master/Recommender%20Systems%20Algorithms.ipynb)
:::
### 對照模型4: IMP_GCN
:::success
#### 說明
[參考資料](https://github.com/liufancs/IMP_GCN)
:::
### 對照模型5: DANSER
:::info
#### 說明
[參考資料](https://github.com/qitianwu/DANSER-WWW-19)
:::
### 對照模型6: LightGCN
:::warning
#### 說明
[參考資料](https://github.com/kuandeng/LightGCN)
:::
### 其他以GNN為基礎的對照模型
:::success
#### 說明
[參考資料](https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems)
:::