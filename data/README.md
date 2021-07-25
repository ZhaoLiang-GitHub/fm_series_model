本仓库中使用的数据是在网上公开数据获得测试数据，与原公司无关，特此说明

在实际工作中，数据基本上是保存在分布式集群如HDFS中，通过分布式框架(例如spark，Hadoop，MapReduce)来处理数据。
在分布式框架处理中将原始数据保存成tfrecords的格式方便接下来在tensorflow中进行处理
```scala
// scala通过spark处理原始数据并保存成tfrecord
// pyspark同理
val sparkSession = createSparkSession(this.getClass.getName)
val orgDf = sparkSession.read.parquet("")
    .select("id","label")
val featureDf = sparkSession.read.parquet("")
    .select("id","feature_1"....,"feature_n")
val resultDf = orgDf.join(featureDf,Seq("id"),"left")
    .select("id","feature_1",...,"feature_n","label")
resultDf.write.mode("overwrite").format("tfrecords").save("hdfs://***/data")
```

```python
# python语言csv转tfrecords
import pandas as pd
import tensorflow as tf

dataframe = pd.read_csv("./**.csv", header=0)  # label,feature_1,...,feature_n
with tf.io.TFRecordWriter("./**.tfrecords") as writer:
    for index in range(dataframe.shape[0]):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[dataframe.iloc[index, 0]])),
                    ...
                    "feature_i": tf.train.Feature(int64_list=tf.train.Int64List(value=[dataframe.iloc[index, i]])),
                }
            )
        )
        writer.write(record=example.SerializeToString())


```
