import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Input
import random
from .config import FeatureConfig, ModelConfig


def data_gen(feature_config: FeatureConfig, model_config: ModelConfig):
    train_data = _data_gen(feature_config.train_data_path, feature_config, model_config)
    val_data = _data_gen(feature_config.val_data_path, feature_config, model_config)
    test_data = _data_gen(feature_config.test_data_path, feature_config, model_config)

    return train_data, val_data, test_data


def _data_gen(path: str, feature_config: FeatureConfig, model_config: ModelConfig, batch_size: int = 128,
              use_hvd: bool = False, use_sample_weigh: bool = False):
    """
    从tfrecords数据中获得tensor
    :param path: 数据路径地址，当有很多个文件时可采用正则表达
    :param feature_config: 特征配置对象
    :param batch_size: 每次返回的tensor中的批次
    :param use_hvd: 是否使用 hrovod 分布式
    :param use_sample_weight: 是否使用样本权重
    :return: 特征tensor ,label ,样本权重(if use_sample_weight)

    Attribute:
        feature_description：每个特征被解析成什么样子的特征，具体解析类型见tf.io
        _parse_function    ：根据feature_description将输入的数据根据列名解析成对应的tensor格式
                             最后删除在模型中不用的特征，否则会warning
    """
    print("data read start for {}".format(path))
    ds_fs = tf.data.Dataset.list_files(file_pattern=path)  # 可以从正则路径中获得文件
    file_name = sorted(ds_fs.as_numpy_iterator())  # 获得绝对路径
    random.shuffle(file_name)
    reader = tf.data.TFRecordDataset(filenames=file_name, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    feature_description = {
        "label": tf.io.FixedLenFeature([1], tf.float32, default_value=0.0),
    }
    for f in feature_config.EMB_COLUMNS:
        feature_description[f] = tf.io.FixedLenFeature([199], tf.float32, default_value=tf.zeros(199, dtype=tf.float32))
    for f in feature_config.NUM_COLUMNS:
        feature_description[f] = tf.io.FixedLenFeature([1], tf.float32, default_value=0.0)
    for f in feature_config.CAT_COLUMNS:
        feature_description[f] = tf.io.FixedLenFeature([1], tf.float32, default_value=0.0)

    def _parse_function(exam_proto):
        """
        映射函数，用于解析一条example
        :param exam_proto: 输入的一条tfrecords
        :return:
        """
        feature = tf.io.parse_single_example(exam_proto, feature_description)
        label = feature["label"]
        feature.pop("label")  # 删除在模型中不用的特征
        return feature, label

    tensor = reader \
        .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(batch_size, drop_remainder=True)
    # print("=" * 60)
    # for line in tensor.take(1):
    #     print(line)
    # print("=" * 60)
    print("data read done")
    return tensor


def define_input_layer(sparse_feature: list, dense_feature: list, cat_id_map: dict, model_config: ModelConfig):
    """
    离散特征向量化层，连续特征全联接层
    """
    sparse_feature_layer = {}
    for f in sparse_feature:
        sparse_feature_layer[f] = Embedding(
            input_dim=cat_id_map[f] + 1 if f in cat_id_map else cat_id_map["default"],
            input_length=1,
            output_dim=model_config.embedding_size,
            embeddings_initializer=model_config.kernel_initializer,
            embeddings_regularizer=model_config.kernel_regularizer,
            name="sparse_embedding_layer_{}".format(f),
        )
        if model_config.model_type == "ifm":
            sparse_feature_layer["IFM_{}".format(f)] = Embedding(
                input_dim=cat_id_map[f] + 1 if f in cat_id_map else cat_id_map["default"],
                input_length=1,
                output_dim=model_config.ifm_hidden_units,
                embeddings_initializer=model_config.kernel_initializer,
                embeddings_regularizer=model_config.kernel_regularizer,
                name="sparse_embedding_layer_for_IFM_{}".format(f),
            )

    dense_feature_layer = {}
    for f in dense_feature:
        dense_feature_layer[f] = Dense(
            units=model_config.embedding_size,
            kernel_initializer=model_config.kernel_initializer,
            kernel_regularizer=model_config.kernel_regularizer,
            activation=None,
            use_bias=False,
            name="dense_feature_layer_{}".format(f)
        )
        if model_config.model_type == "ifm":
            dense_feature_layer["IFM_{}".format(f)] = Dense(
                units=model_config.ifm_hidden_units,
                kernel_initializer=model_config.kernel_initializer,
                kernel_regularizer=model_config.kernel_regularizer,
                activation=None,
                use_bias=False,
                name="dense_feature_layer_for_IFM_{}".format(f)
            )

    return sparse_feature_layer, dense_feature_layer


def define_input_tensor(sparse_feature: list, dense_feature: list, emb_feature: list, model_config: ModelConfig):
    """
    定义输入层
    """
    input, sparse_feature_input, dense_feature_input, emb_feature_input = {}, {}, {}, {}
    for f in sparse_feature:
        sparse_feature_input[f] = Input(shape=(1,), name=f, dtype=tf.float32)
        if model_config.model_type == "ifm":
            sparse_feature_input["IFM_{}".format(f)] = Input(shape=(1,), name="IFM_{}".format(f), dtype=tf.float32)
    for f in emb_feature:
        emb_feature_input[f] = Input(shape=(199,), name=f, dtype=tf.float32)
    for f in dense_feature:
        dense_feature_input[f] = Input(shape=(1,), name=f, dtype=tf.float32)
        if model_config.model_type == "ifm":
            dense_feature_input["IFM_{}".format(f)] = Input(shape=(1,), name="IFM_{}".format(f), dtype=tf.float32)
    input.update(sparse_feature_input)
    input.update(emb_feature_input)
    input.update(dense_feature_input)
    return input, sparse_feature_input, dense_feature_input, emb_feature_input
