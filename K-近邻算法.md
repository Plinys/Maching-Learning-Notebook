# K-近邻算法
**工作原理：** 训练样本集中每个数据都存在对应的标签，输入没有标签的新数据后,将新数据的每个特征与样本数据集中相应特征进行比较，
然后算法提取集中特征最相似数据的分类标签（前K个），选取其中出现概率最高的标签作为结果，称为K近邻算法。
## 算法详解
- 计算数据集中点与当前输入点的距离
- 按距离递增次序排序
- 选取距离最小的K个点
- 确定前K个点所在类别的出现概率
- 返回前K个点出现频率最高的类别标签作为当前预测分类
## Python 实现K-近邻算法
```
  def classify(inX, dataSet, labels, k)
      dateSetSize = dataSet.shape[0]
      diffMat = numpy.tile(inX, (dataSetSize,1)) - dataSet
      # tile函数用来重复构造指定新数组
      sqDiffMat = diffMat ** 2
      sqDistances = sqDiffMat.sum(axis=1)
      distances = sqDistances ** 0.5
      sortedDistIndices = distances.argsort() #argsort函数返回的是数组值从小到大的索引值
      classCount = {}
      for i in range(k):
        votelabels = labels[sortedDistIndices[i]]
        classCount[votelabels] = classCount.get(voteLabel, 0) + 1
      sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
      return sorteedClassCount[0][0]
```
## 数据归一化
![归一化方法](https://github.com/Plinys/Maching-Learning-Notebook/blob/master/picture/%E5%BD%92%E4%B8%80%E5%8C%96.png)  
以极差变换法为例:  
newValue = (OldValue-min)/(max-min)  

```
  def autoNorm(dataSet）：
      minVals = dataSet.min(0)
      maxVals = dataSet.amx(0)
      ranges = maxVals - minVals
      normDataSet = zeros(shape(dataSet))
      m = dataSet.shape[0]
      normDataSet = dataSet - tile(minVals, (m,1))
      normDataSet = normDataSet/tile(ranges, (m, 1))
      return normDataSet, ranges, minVals
```
