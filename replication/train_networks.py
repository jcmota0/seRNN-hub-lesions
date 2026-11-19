"""Train my own seRNN and L1 networks with the authors' code, to compare with theirs.

Every network uses one of the authors' cost strengths, so each has a "twin" in their
data with the same strength:
  L1:    np.linspace(1e-7, 0.03, 1001)   (read from their network_data.mat)
  seRNN: np.linspace(0.001, 0.3, 1001)   (read from their network_data.mat)
By default it trains every 10th one: networks 1, 11, 21, ..., 1001.
Their notebook text gives a different seRNN list, np.linspace(1e-6, 1, 1001);
--strengths notebook uses that one instead, to test which list they really trained with.

Training settings are the authors' (seRNN_demo.ipynb and recording_decoding_demo.ipynb):
100 ReLU units on a 5x5x4 grid, input noise 0.05, Adam, 10 epochs,
5,120 training problems in batches of 128.

Each network is saved as a MATLAB file in the output folder, with the same layout as
the authors' data, so their analysis code can read it:
  w_rec     11 x 100 x 100  recurrent weights before training and after each epoch
  accuracy  11 x 2          accuracy (training, validation) at the same 11 points
  group, network, strength, strength_list, seed

Usage:
  python replication/train_networks.py --group seRNN
  python replication/train_networks.py --group L1 --networks 1 11 21
Networks that are already saved are skipped, so a stopped run can be restarted.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.io
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from authors_code import RNNWeightMatrixHistoryI, SE1_sWc, mazeGeneratorI  # noqa: E402

STRENGTHS = {
    # what the authors' saved data (network_data.mat) records
    "data": {
        "L1": np.linspace(1e-7, 0.03, 1001),
        "seRNN": np.linspace(0.001, 0.3, 1001),
    },
    # what the text of their recording_decoding_demo.ipynb says
    "notebook": {
        "L1": np.linspace(1e-7, 0.03, 1001),
        "seRNN": np.linspace(1e-6, 1, 1001),
    },
}
DEFAULT_OUT = {"data": "~/seRNN-runs", "notebook": "~/seRNN-runs-notebook"}
SEED_OFFSET = {"L1": 0, "seRNN": 10000}  # so L1 network 5 and seRNN network 5 get different seeds


def train_one(group, network, out_dir, strength_list):
    out_file = os.path.join(out_dir, f"{group}_{network:04d}.mat")
    if os.path.exists(out_file):
        print(f"{group} {network}: already saved, skipping")
        return

    strength = float(STRENGTHS[strength_list][group][network - 1])
    seed = SEED_OFFSET[group] + network

    # Everything below follows the authors' notebooks; only the seed handling is ours.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TensorFlow

    gen = mazeGeneratorI(goal_presentation_steps=20, delay_steps=10, choices_presentation_steps=20)
    tfdf = gen.construct_tf_data(number_of_problems=5120, batch_size=128)
    tfdf_test = gen.construct_tf_data(number_of_problems=2560, batch_size=128)
    example_data = next(iter(tfdf))

    if group == "seRNN":
        regu = SE1_sWc(se1=strength)
    else:
        regu = tf.keras.regularizers.L1(strength)

    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(stddev=0.05),
        tf.keras.layers.SimpleRNN(100, activation='relu', recurrent_initializer='orthogonal',
                                  return_sequences=False, recurrent_regularizer=regu),
        tf.keras.layers.Dense(4, activation='softmax'),
    ])
    tf_model.build(input_shape=example_data[0].shape)
    tf_model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    # Accuracy before training, so there are 11 time points like the authors' data
    _, acc_train_0 = tf_model.evaluate(tfdf, verbose=0)
    _, acc_val_0 = tf_model.evaluate(tfdf_test, verbose=0)

    t0 = time.time()
    history = tf_model.fit(tfdf, epochs=10, validation_data=tfdf_test, verbose=0,
                           callbacks=RNNWeightMatrixHistoryI(RNN_layer_number=1))
    seconds = time.time() - t0

    w_rec = np.stack(history.history['RNN_Weight_Matrix'])  # 11 x 100 x 100
    accuracy = np.column_stack([
        [acc_train_0] + history.history['accuracy'],
        [acc_val_0] + history.history['val_accuracy'],
    ])  # 11 x 2

    scipy.io.savemat(out_file, {
        "group": group, "network": network, "strength": strength, "strength_list": strength_list, "seed": seed,
        "w_rec": w_rec, "accuracy": accuracy,
    })
    print(f"{group} {network:4d}  strength {strength:.6g}  final validation accuracy "
          f"{accuracy[-1, 1]:.3f}  total weight {np.abs(w_rec[-1]).sum():7.1f}  ({seconds:.0f} s)",
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True, choices=["seRNN", "L1"])
    p.add_argument("--networks", type=int, nargs="+", default=list(range(1, 1002, 10)),
                   help="which of the authors' 1,001 networks to match (default: every 10th)")
    p.add_argument("--strengths", choices=["data", "notebook"], default="data",
                   help="which list of cost strengths to use: the authors' saved data (default) "
                        "or the text of their notebook; they differ for seRNN")
    p.add_argument("--out", default=None,
                   help="folder for the saved networks (default: ~/seRNN-runs for 'data', "
                        "~/seRNN-runs-notebook for 'notebook', both outside the repo)")
    args = p.parse_args()

    out = os.path.expanduser(args.out or DEFAULT_OUT[args.strengths])
    os.makedirs(out, exist_ok=True)
    for network in args.networks:
        train_one(args.group, network, out, args.strengths)


if __name__ == "__main__":
    main()
