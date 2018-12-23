# NLP 分类

## 1. 模型概览


| models | tensorflow | pytorch |
| - | - | - |
|  fastText  |   Yes   |   Yes    |
|  TextCNN  |   Yes   |   Yes    |
|  TextRNN  |   Yes   |   Yes    |
|  TextRCNN  |   Yes   |   False    |



## 2. 数据集及数据处理
* 数据： IMDB
* 数据分割：1w, 2w, 3w ...   (未实现)


## 3. tensorflow实验记录
- [x] ** fastText模型 **

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 128 | 0.717 | 0.839 | - |


- [x] ** TextCNN模型 **

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 128 | 0.537 | 0.878 | - |


- [x] ** TextRNN模型 **
<font color="red">过拟合：  Train Loss:0.184	Train Accuracy:0.932</font>

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 128 | 0.417 | 0.843 | - |

dropout=0.5:  过拟合效果降低，准确率也有所降低
epoch=5:
Train Loss:0.303	Train Accuracy:0.883
Validation Loss:0.425	Validation Accuracy: 0.824


- [x] ** TextRCNN模型 **  
* BiRNN
<font color="red">不建议考虑该模型</font>

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 128 | 0.444 | 0.879 | - |



## 4. pytorch实验记录
- [x] ** fastText模型 **

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 128 | 0.07620274275541306 | 0.85 | - |

- [x] ** TextCNN模型 **

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 50 | 0.0011307994136586785 | 0.8 | - |


- [x] ** TextRNN模型 **

| 训练数据大小 | batch_size | loss | acc | 运行时间 |
| - | - | - | - | - |
| 2.5w | 400 | - | - | - |