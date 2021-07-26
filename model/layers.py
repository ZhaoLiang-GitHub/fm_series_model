from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout, Softmax, Flatten, Concatenate
from tensorflow.keras.activations import linear, relu
import tensorflow as tf
import itertools
from config import ModelConfig


class DNN(Layer):
    def __init__(self, model_config: ModelConfig, **kwargs):
        self.config = model_config
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = BatchNormalization()
        self.dropout = Dropout(self.config.dropout_rate, trainable=True)
        self.dnn_dense = []
        for i in self.config.dnn_dim:
            self.dnn_dense.append(Dense(
                units=i,
                kernel_initializer=self.config.kernel_initializer,
                kernel_regularizer=self.config.kernel_regularizer,
                activation=relu,
                use_bias=True,
                bias_initializer=self.config.bias_initializer,
                name="dnn_layer_{}".format(i),
            ))
        self.output_dense = Dense(
            units=1,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            name="dnn_output",
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        inputs = self.bn(inputs)
        for _ in range(len(self.config.dnn_dim)):
            inputs = self.dropout(inputs)
            inputs = self.dnn_dense[_](inputs)
        output = self.output_dense(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return None, 1


class FM_interaction(Layer):
    """
    fm交叉层
    输入的张量尺寸是 [batch_size,feature_num,embedding_size]
    输出的张量尺寸是 [batch_size,feature_num*(feature_num-1)/2,embedding_size]
    即输入的特征两两交叉的原始结果，交叉的计算方法为哈德玛积
    """

    def __init__(self, **kwargs, ):
        super(FM_interaction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        row = []
        col = []
        for f1, f2 in itertools.combinations(inputs, 2):
            row.append(f1)
            col.append(f2)
        p = Concatenate(axis=1)(row)
        q = Concatenate(axis=1)(col)
        linear_output = p * q
        return linear_output

    def compute_output_shape(self, input_shape):
        return None, input_shape[1] * (input_shape[1] - 1) / 2, input_shape[2]


class FM_pool(Layer):
    """
    fm pool层
    输入的张量尺寸是 [batch_size,feature_num,embedding_size]
    输出的张量尺寸是 [batch_size,1,embedding_size]
    即将 FM_interaction 输出的结果进行pooling
    """

    def __init__(self, **kwargs):
        super(FM_pool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        """
        将FM_interaction的结果进行pooling时间复杂度是o(N**2)
        在FM的原始论文中通过改写将时间负责降低至O(k*N)
        """
        # pooling_output = tf.reduce_sum(FM_interaction()(inputs), axis=1, keepdims=True)  # O(k*N*N)
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        pooling_output = 0.5 * (square_of_sum - sum_of_square)
        return pooling_output

    def compute_output_shape(self, input_shape):
        return None, 1, input_shape[2]


class FM_num(Layer):
    """
    FM 最终计算层
    输入的张量尺寸是 [batch_size,feature_num,embedding_size]
    输出的张量尺寸是 [batch_size,1]
    即将FM_pool得到的向量结果直接求和，得到最终的fm输出结果
    """

    def __init__(self, **kwargs):
        super(FM_num, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        num_output = tf.reduce_sum(FM_pool()(inputs), axis=2, keepdims=False)
        return num_output

    def compute_output_shape(self, input_shape):
        return None, 1


class AFM_interaction(Layer):
    """
    AFM
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        self.config = model_config
        super(AFM_interaction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True
        self.W = Dense(
            units=self.config.afm_attention_units,
            activation=relu,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            use_bias=True,
            bias_initializer=self.config.bias_initializer,
        )
        self.h = Dense(
            units=1,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
        )

    def call(self, inputs, *args, **kwargs):
        interaction = FM_interaction()(inputs)
        attention_score = Softmax()(self.h(self.W(interaction)))
        afm_out = interaction * attention_score

        return afm_out

    def compute_output_shape(self, input_shape):
        return None, input_shape[1] * (input_shape[1] - 1) / 2, input_shape[2]


class AFM_num(Layer):
    def __init__(self, model_config: ModelConfig, **kwargs):
        self.config = model_config
        super(AFM_num, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = Dense(
            units=self.config.afm_attention_units,
            activation=relu,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            use_bias=True,
            bias_initializer=self.config.bias_initializer,
            name="afm_w",
        )
        self.h = Dense(
            units=1,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            name="afm_h",
        )
        self.p = Dense(
            units=1,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            use_bias=False,
            name="afm_p",
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        interaction = FM_interaction()(inputs)
        attention_score = Softmax()(self.h(self.W(interaction)))
        output = self.p(tf.reduce_sum(attention_score * interaction, axis=1, ))
        return output

    def compute_output_shape(self, input_shape):
        return None, 1


class NFM(Layer):
    def __init__(self, model_config: ModelConfig, **kwargs):
        self.config = model_config
        super(NFM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True
        self.bn = BatchNormalization()
        self.dropout = Dropout(self.config.dropout_rate, trainable=True)
        self.dnn_dense = []
        for i in self.config.dnn_dim:
            self.dnn_dense.append(Dense(
                units=i,
                kernel_initializer=self.config.kernel_initializer,
                kernel_regularizer=self.config.kernel_regularizer,
                activation=relu,
                use_bias=True,
                bias_initializer=self.config.bias_initializer,
                name="dnn_layer_{}".format(i),
            ))
        self.output_dense = Dense(
            units=1,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            name="dnn_output",
        )

    def call(self, inputs, *args, **kwargs):
        interaction = FM_interaction()(inputs)
        linear_input = self.bn(Flatten()(interaction))
        for _ in range(len(self.config.dnn_dim)):
            linear_input = self.dropout(linear_input)
            linear_input = self.dnn_dense[_](linear_input)
        nfm_out = self.output_dense(linear_input)
        return nfm_out

    def compute_output_shape(self, input_shape):
        return None, 1


class IFM(Layer):
    def __init__(self, model_config: ModelConfig, **kwargs):
        self.config = model_config
        super(IFM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True
        self.field_feature_score = Dense(
            units=self.config.embedding_size,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            use_bias=False,
            trainable=True,
        )
        self.W = Dense(
            units=self.config.afm_attention_units,
            activation=relu,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
            use_bias=True,
            bias_initializer=self.config.bias_initializer,
        )
        self.h = Dense(
            units=1,
            activation=linear,
            kernel_initializer=self.config.kernel_initializer,
            kernel_regularizer=self.config.kernel_regularizer,
        )


    def call(self, inputs, *args, **kwargs):
        interaction = FM_interaction()(inputs[0])
        attention_score = Softmax()(self.h(self.W(interaction)))
        afm_out = interaction * attention_score
        field_feature_score = self.field_feature_score(FM_interaction()(inputs[1]))
        ifm_output = tf.reduce_sum(tf.reduce_sum(afm_out * field_feature_score, axis=-1), axis=-1, keepdims=True)
        return ifm_output

    def compute_output_shape(self, input_shape):
        return None, 1
