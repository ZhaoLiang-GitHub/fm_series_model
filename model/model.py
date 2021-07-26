from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from .util import define_input_tensor, define_input_layer
from config import FeatureConfig
from .layers import *


def FM_series_models(feature_config: FeatureConfig, model_config: ModelConfig):
    sparse_feature, dense_feature, emb_feature = feature_config.CAT_COLUMNS, feature_config.NUM_COLUMNS, feature_config.EMB_COLUMNS

    sparse_embed_layers, dense_dense_layers = define_input_layer(sparse_feature,
                                                                 dense_feature,
                                                                 feature_config.cat_id_map,
                                                                 model_config
                                                                 )
    # input layer
    input, sparse_feature_input, dense_feature_input, emb_feature_input = define_input_tensor(sparse_feature,
                                                                                              dense_feature,
                                                                                              emb_feature,
                                                                                              model_config,
                                                                                              )

    # embedding layer
    embedding = []  # 离散特征+数值特征向量化
    IFM_embedding = []  # ifm 模型中一个特征有两个向量表示，需要分开记录
    for f in sparse_feature:
        embedding.append(
            Reshape((1, model_config.embedding_size))(sparse_embed_layers[f](sparse_feature_input[f]))
        )
        if "ifm" in model_config.model_type:
            IFM_embedding.append(
                Reshape((1, model_config.ifm_hidden_units))(
                    sparse_embed_layers["IFM_{}".format(f)](sparse_feature_input[f]))
            )
    for f in dense_feature:
        embedding.append(
            Reshape((1, model_config.embedding_size))(dense_dense_layers[f](dense_feature_input[f]))
        )
        if "ifm" in model_config.model_type:
            IFM_embedding.append(
                Reshape((1, model_config.ifm_hidden_units))(
                    dense_dense_layers["IFM_{}".format(f)](dense_feature_input[f]))
            )

    embedding_concat = Concatenate(axis=1, name="embedding")(embedding)
    linear_input = Flatten()(embedding_concat)
    all_input = Flatten()(Concatenate(axis=1, name='all_feature')(list(emb_feature_input.values()) + [linear_input]))

    # 全部特征lr的结果
    all_lr = Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=model_config.kernel_initializer,
        kernel_regularizer=model_config.kernel_regularizer,
        use_bias=True,
        bias_initializer=model_config.bias_initializer,
        name="lr_fc",
    )(BatchNormalization()(all_input))

    # 离散特征+数值特征FM的结果
    feature_embed_fm = FM_num()(embedding_concat)
    # 全部特征l拉平后输入dnn
    all_dnn = DNN(model_config)(all_input)
    # 离散特征+数值特征交叉得到1个特征输入到dnn
    feature_embed_fm_dnn = DNN(model_config)(Flatten()(FM_interaction()(embedding)))
    # 离散特征+数值特征交叉得到1个特征输入到dnn
    feature_embed_fm_pool_dnn = DNN(model_config)(FM_pool()(embedding))
    # 离散特征输入afm
    feature_embed_afm = AFM_num(model_config)(embedding)
    # 离散特征+数值特征输入NFM层
    feature_embed_nfm = NFM(model_config)(embedding)

    if model_config.model_type == "lr":
        output_list = [all_lr]
    elif model_config.model_type == "dnn":
        output_list = [all_dnn]
    elif model_config.model_type == "fm":
        output_list = [all_lr, feature_embed_fm]
    elif model_config.model_type == "deepfm":
        output_list = [all_lr, feature_embed_fm, all_dnn]
    elif model_config.model_type == "afm":
        output_list = [feature_embed_afm]
    elif model_config.model_type == "deepafm":
        output_list = [all_lr, feature_embed_afm, all_dnn]
    elif model_config.model_type == "nfm":
        output_list = [all_lr, feature_embed_nfm]
    elif model_config.model_type == "deepnfm":
        output_list = [all_lr, feature_embed_nfm, all_dnn]
    elif model_config.model_type == "ifm":
        # ifm 层输出结果
        feature_embed_ifm = IFM(model_config)([embedding] + [IFM_embedding])
        output_list = [feature_embed_ifm, all_lr]
    elif model_config.model_type == "deepifm":
        feature_embed_ifm = IFM(model_config)([embedding] + [IFM_embedding])
        output_list = [feature_embed_ifm, all_lr, all_dnn]
    else:
        raise ValueError("输入类型错误🙅")
    output = Dense(
        units=1,
        activation=tf.keras.activations.sigmoid,
        kernel_initializer=model_config.kernel_initializer,
        name="output",
    )(tf.reduce_sum(output_list, axis=0, ))
    model = Model(inputs=input, outputs=output)
    return model
