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
        self.model_type = "deepfm"

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
        self.ckpt = "./data/ckpt"
        self.tb = "./data/tb"
        self.afm_attention_units = (self.embedding_size // 5) + 1

        self.nfm_fm_type = "fm"

        self.ifm_hidden_units = (self.embedding_size // 5) + 1
        # self.params = {
        #     'k': 8,
        #     'dnn_dim': [256, 128, 64],  # [256, 128, 64]
        #     'dnn_dr': 0.5,
        #     'epochs': 50,
        #     'batch_size': 1024 * 4,
        #     'feat_style': 'std',
        #     'attention_dr': 0.5,  # 0.2
        #     'lr': 1e-3,  # 1e-3
        #     'field_interaction_factor': 16,
        #     'sampling': 0.1,
        #     'threshold': 6000,  # num of batches
        #     'use_weight': True,
        #     'use_hvd': True,
        #     'model_name': "theme-mmoe-page-theme"
        # }
        #
        # self.use_hvd = self.params['use_hvd']  # use parallel training or not, set to False if running local test
        # job_dir = self.params["model_name"]
