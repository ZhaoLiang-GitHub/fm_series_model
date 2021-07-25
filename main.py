import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model.config import FeatureConfig, ModelConfig
from model.util import data_gen
from model.model import FM_series_models


def run(debug=False):
    feature_config = FeatureConfig(debug)
    model_config = ModelConfig(debug)
    train_data, val_data, test_data = data_gen(feature_config, model_config)
    model = FM_series_models(feature_config, model_config)
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(
        optimizer=Adam(model_config.learning_rate),
        loss=binary_crossentropy,
        metrics=[Accuracy(), AUC(), Precision(), Recall(), ],
    )
    history = model.fit(
        test_data,
        validation_data=val_data,
        epochs=model_config.epochs,
        verbose=model_config.verbose,
        batch_size=model_config.batch_size,
        callbacks=[
            EarlyStopping(monitor="loss", mode="min", min_delta=0.0005, patience=3, restore_best_weights=True, ),
            ModelCheckpoint(monitor="loss", mode="min", filepath=os.path.join(model_config.ckpt, 'cp-{epoch:04d}.ckpt'),
                            save_best_only=True, ),
            TensorBoard(log_dir=model_config.tb, write_graph=True, write_images=True, ),
        ]
    )
    model.save(os.path.join(model_config.ckpt, './model'), save_format='tf')
    metrics = model.evaluate(test_data)
    print_metrics(ckpt_dir=model_config.ckpt, metrics_names=model.metrics_names, metrics=metrics, history=history)


def print_metrics(ckpt_dir, metrics_names, metrics, history):
    best_epoch = int(
        os.path.splitext(sorted(list(filter((lambda i: "ckpt" in i), os.listdir(ckpt_dir))))[-1])[0].split('-')[1]
    ) - 1
    metrics_train = [history.history[metric_name][best_epoch] for metric_name in metrics_names]
    metrics_val = [history.history['val_{}'.format(metric_name)][best_epoch] for metric_name in metrics_names]
    star_num = 120
    print('*' * star_num)
    print('*' * star_num)
    header = ['data set'] + metrics_names
    fmt_header = '|' + '|'.join(['{:>16}'] * len(header)) + '|'
    fmt_values = '|{:>16}|' + '|'.join(['{:16.4f}'] * len(metrics_names)) + '|'
    print(fmt_header.format(*header))
    for ds_name, values in zip(['train', 'val', 'test'], [metrics_train, metrics_val, metrics]):
        print(fmt_values.format(ds_name, *values))
    print('*' * star_num)
    print('*' * star_num)


def predict():
    model = tf.keras.models.load_model("")
    pass


if __name__ == '__main__':
    run(True)
