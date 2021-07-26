from tensorflow.keras.initializers import glorot_normal, Zeros, random_normal
from tensorflow.keras.regularizers import l2


class FeatureConfig(object):
    def __init__(self, debug):
        """
        有关数据特征的相关配置
        Attribute:
            NUM_COLUMNS : 连续型数值特征，后续会进入dense层生成向量
            EMB_COLUMNS : 向量特征，每个特征是一个固定长度的数组
            CAT_COLUMNS : 离散特征，后续会经过embedding层生成向量
            cat_id_map  : 离散特征可枚举值数组
        """
        self.NUM_COLUMNS = ["num_feature_1", "num_feature_2", ]
        self.EMB_COLUMNS = ["emb_feature_1", "emb_feature_2", ]
        self.CAT_COLUMNS = ["cat_feature_1", "cat_feature_2", ]
        self.cat_id_map = {"cat_feature_1": 9, "cat_feature_2": 9, "default": 10}
        self.label = 'label'
        if debug:  # 本地测试
            self.train_data_path = self.val_data_path = self.test_data_path = "./data/part*"
        else:  # 集群训练模型使用hdfs地址
            self.train_data_path = 'hdfs://***/train/part*'
            self.val_data_path = 'hdfs://***/val/part*'
            self.test_data_path = 'hdfs://***/test/part*'


class ModelConfig(object):
    def __init__(self, debug):
        """
        有关模型参数的相关配置
        Attribute

        """
        self.model_type = "deepifm"
        self.ckpt = "./data/ckpt"
        self.tb = "./data/tb"

        self.learning_rate = 0.01
        self.batch_size = 1024
        self.epochs = 1
        self.embedding_size = 8
        self.verbose = 1
        self.seed = 1024
        self.kernel_initializer = glorot_normal(self.seed)
        self.kernel_regularizer = l2(0.01)
        self.bias_initializer = random_normal()
        self.dnn_dim = [128, 64, 32]
        self.dropout_rate = 0.2
        self.afm_attention_units = (self.embedding_size // 5) + 1
        self.ifm_hidden_units = (self.embedding_size // 5) + 1
