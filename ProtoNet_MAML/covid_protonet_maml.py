from google.colab import drive
import os

def mountGoogleDrive(myFolder):
    root = '/content/drive'
    drive.mount(root, force_remount=True)
    dest_folder = root + '/My Drive' + myFolder
    os.chdir(dest_folder)
    curr_path = os.getcwd()
    if len(curr_path) > 0:
        print('Current path: ', os.getcwd())
        return curr_path
    else:
        raise Exception('\nCannot mount the Google Drive\n')

myFolder = '/ICLR 2026/'
curr_path = mountGoogleDrive(myFolder)

import math
import random
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers, Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten

warnings.filterwarnings('ignore')

base_path = '/content/drive/My Drive/ICLR 2026/Datasets/COVID/'
trails = range(1, 6)
train_datasets = {}
test_datasets = {}

for trail in trails:
    train_file = f'COVID-19_train_trail_{trail}.csv'
    test_file = f'COVID-19_test_trail_{trail}.csv'
    train_datasets[trail] = pd.read_csv(base_path + train_file)
    test_datasets[trail] = pd.read_csv(base_path + test_file)



def computeMeasure(class_num, predicted_label, true_label):

    # Compute the confusion matrix
    cnf_matrix = confusion_matrix(true_label, predicted_label)

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Convert to float for more precise calculations
    FP, FN, TP, TN = map(lambda x: x.astype(float), [FP, FN, TP, TN])

    # Initialize variables to store metrics
    # Use np.errstate to ignore divide by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = np.nan_to_num(TP / (TP + FN))

        # Specificity or true negative rate
        TNR = np.nan_to_num(TN / (TN + FP))

        # Precision or positive predictive value
        PPV = np.nan_to_num(TP / (TP + FP))

        # Negative predictive value
        NPV = np.nan_to_num(TN / (TN + FN))

        # False positive rate
        FPR = np.nan_to_num(FP / (FP + TN))

        # False negative rate
        FNR = np.nan_to_num(FN / (TP + FN))

        # False discovery rate
        FDR = np.nan_to_num(FP / (TP + FP))

        # F1 score
        F_1 = np.nan_to_num(2 * (PPV * TPR) / (PPV + TPR))

        # Per-class accuracy
        ACC_Class = np.nan_to_num((TP + TN) / (TP + FP + FN + TN))

        # Overall accuracy
        ACC = np.sum(np.diag(cnf_matrix)) / cnf_matrix.sum()

    # Compute discriminative power index for all classes
    d_idx_vector = np.log2(1 + ACC) + np.log2(1 + (TPR + TNR) / 2)
    d_idx = d_idx_vector.mean() # do average

    # Prepare and return results
    results = [
        d_idx,
        ACC,
        TPR.mean(),
        TNR.mean(),
        PPV.mean(),
        NPV.mean()
    ]

    return results

def prototypical_loss(emb_support, emb_query, n_way, k_shot):

    prototypes = []
    for j in range(n_way):
        start = j * k_shot
        end = start + k_shot
        proto = tf.reduce_mean(emb_support[start:end], axis=0)
        prototypes.append(proto)
    prototypes = tf.stack(prototypes, axis=0)

    q_query = tf.shape(emb_query)[0] // n_way
    labels = []
    for j in range(n_way):
        labels.append(tf.fill([q_query], j))
    labels = tf.concat(labels, axis=0)

    qe = tf.expand_dims(emb_query, axis=1)
    pr = tf.expand_dims(prototypes, axis=0)
    dists = tf.reduce_sum(tf.square(qe - pr), axis=2)
    logits = -dists

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
    )
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
    return loss, acc


def train_protonet_cnn(
    X_train,
    y_train,
    embedding_dim=32,
    n_way=5,
    k_shot=5,
    q_query=10,
    meta_epochs=20,
    episodes_per_epoch=20,
    learning_rate=1e-3,
):
    classes = np.unique(y_train)
    input_shape = (X_train.shape[1], X_train.shape[2])

    backbone = Sequential(
        [
            Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape),
            GlobalAveragePooling1D(),
            Dense(embedding_dim, activation=None),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    X_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)

    for epoch in range(meta_epochs):
        epoch_losses = []
        epoch_accs = []

        for epi in range(episodes_per_epoch):
            if len(classes) < n_way:
                continue

            episode_classes = np.random.choice(classes, size=n_way, replace=False)

            support_idxs = []
            query_idxs = []
            valid_episode = True

            for c in episode_classes:
                idx_c = np.where(y_train == c)[0]
                if len(idx_c) < (k_shot + q_query):
                    valid_episode = False
                    break
                chosen = np.random.choice(
                    idx_c, size=(k_shot + q_query), replace=False
                )
                support_idxs.extend(chosen[:k_shot])
                query_idxs.extend(chosen[k_shot:k_shot + q_query])

            if not valid_episode or len(support_idxs) < n_way * k_shot:
                continue

            support_idxs = np.array(support_idxs)
            query_idxs = np.array(query_idxs)

            X_support = tf.gather(X_tf, support_idxs, axis=0)
            X_query = tf.gather(X_tf, query_idxs, axis=0)

            with tf.GradientTape() as tape:
                emb_support = backbone(X_support, training=True)
                emb_query = backbone(X_query, training=True)
                loss, acc = prototypical_loss(
                    emb_support, emb_query, n_way, k_shot
                )

            grads = tape.gradient(loss, backbone.trainable_variables)
            optimizer.apply_gradients(zip(grads, backbone.trainable_variables))

            epoch_losses.append(loss.numpy())
            epoch_accs.append(acc.numpy())

        if (epoch + 1) % 5 == 0 and epoch_losses:
            print(
                f"[ProtoNet-CNN] Epoch {epoch + 1}/{meta_epochs} "
                f"loss={np.mean(epoch_losses):.4f}, acc={np.mean(epoch_accs):.4f}"
            )

    return backbone


def predict_protonet_cnn(backbone, X_train, y_train, X_test, k_shot_eval=5):
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

    train_embeds = backbone(X_train_tf, training=False).numpy()
    test_embeds = backbone(X_test_tf, training=False).numpy()

    classes = np.unique(y_train)
    prototypes = []
    for c in classes:
        idx_c = np.where(y_train == c)[0]
        if len(idx_c) < k_shot_eval:
            chosen = idx_c
        else:
            chosen = np.random.choice(idx_c, size=k_shot_eval, replace=False)
        proto = train_embeds[chosen].mean(axis=0)
        prototypes.append(proto)
    prototypes = np.stack(prototypes, axis=0)

    preds = []
    for e in test_embeds:
        dists = np.sum((prototypes - e) ** 2, axis=1)
        c = classes[np.argmin(dists)]
        preds.append(c)
    return np.array(preds)




class SimpleConvNet(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        init = tf.initializers.GlorotUniform()
        self.W1 = tf.Variable(init(shape=(3, 1, 32)), name="W1")
        self.b1 = tf.Variable(tf.zeros([32]), name="b1")
        self.W2 = tf.Variable(init(shape=(3, 32, 32)), name="W2")
        self.b2 = tf.Variable(tf.zeros([32]), name="b2")

    @property
    def trainable_variables(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x, weights=None, training=False):
        if weights is None:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        else:
            W1, b1, W2, b2 = weights
        x = tf.nn.conv1d(x, W1, stride=1, padding="SAME")
        x = tf.nn.bias_add(x, b1)
        x = tf.nn.relu(x)
        x = tf.nn.conv1d(x, W2, stride=1, padding="SAME")
        x = tf.nn.bias_add(x, b2)
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, axis=1)
        return x


def train_maml_cnn(
    X_train,
    y_train,
    num_classes=None,
    meta_epochs=20,
    tasks_per_epoch=20,
    n_way=5,
    k_shot=5,
    q_query=10,
    inner_lr=5e-3,
    outer_lr=1e-3,
    inner_steps=3,
):
    classes = np.unique(y_train)

    maml_model = SimpleConvNet()
    outer_opt = tf.keras.optimizers.Adam(learning_rate=outer_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    X_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)

    feat_dim = 32

    for epoch in range(meta_epochs):
        epoch_losses = []

        for t in range(tasks_per_epoch):
            if len(classes) < n_way:
                continue

            task_classes = np.random.choice(classes, size=n_way, replace=False)

            support_idx_list = []
            query_idx_list = []
            valid_task = True

            for c in task_classes:
                idx_c = np.where(y_train == c)[0]
                if len(idx_c) < (k_shot + q_query):
                    valid_task = False
                    break
                chosen = np.random.choice(idx_c, size=(k_shot + q_query), replace=False)
                support_idx_list.append(chosen[:k_shot])
                query_idx_list.append(chosen[k_shot:k_shot + q_query])

            if not valid_task or len(support_idx_list) < n_way:
                continue

            support_idx = np.concatenate(support_idx_list, axis=0)
            query_idx = np.concatenate(query_idx_list, axis=0)

            X_s = tf.gather(X_tf, support_idx, axis=0)
            y_s_global = tf.gather(y_tf, support_idx, axis=0)
            X_q = tf.gather(X_tf, query_idx, axis=0)
            y_q_global = tf.gather(y_tf, query_idx, axis=0)

            class_to_local = {int(c): i for i, c in enumerate(task_classes)}
            y_s = tf.convert_to_tensor(
                [class_to_local[int(c)] for c in y_s_global.numpy()],
                dtype=tf.int32,
            )
            y_q = tf.convert_to_tensor(
                [class_to_local[int(c)] for c in y_q_global.numpy()],
                dtype=tf.int32,
            )

            with tf.GradientTape() as outer_tape:
                fast_weights = [tf.identity(w) for w in maml_model.trainable_variables]
                head_W = tf.Variable(
                    tf.random.normal([feat_dim, n_way]) * 0.01
                )
                head_b = tf.Variable(tf.zeros([n_way]))

                for _ in range(inner_steps):
                    with tf.GradientTape() as inner_tape:
                        emb_s = maml_model(X_s, weights=fast_weights, training=True)
                        logits_s = tf.matmul(emb_s, head_W) + head_b
                        loss_s = loss_fn(y_s, logits_s)

                    vars_inner = fast_weights + [head_W, head_b]
                    grads_inner = inner_tape.gradient(loss_s, vars_inner)

                    grads_body = grads_inner[: len(fast_weights)]
                    gW_head, gB_head = grads_inner[len(fast_weights) :]

                    grads_body = [
                        tf.stop_gradient(g) if g is not None else None
                        for g in grads_body
                    ]
                    if gW_head is not None:
                        gW_head = tf.stop_gradient(gW_head)
                    if gB_head is not None:
                        gB_head = tf.stop_gradient(gB_head)

                    fast_weights = [
                        w - inner_lr * g if g is not None else w
                        for w, g in zip(fast_weights, grads_body)
                    ]
                    if gW_head is not None:
                        head_W = head_W - inner_lr * gW_head
                    if gB_head is not None:
                        head_b = head_b - inner_lr * gB_head

                emb_q = maml_model(X_q, weights=fast_weights, training=True)
                logits_q = tf.matmul(emb_q, head_W) + head_b
                loss_q = loss_fn(y_q, logits_q)

            grads_outer = outer_tape.gradient(
                loss_q, maml_model.trainable_variables
            )
            grads_vars = [
                (g, v)
                for g, v in zip(
                    grads_outer, maml_model.trainable_variables
                )
                if g is not None
            ]

            if grads_vars:
                outer_opt.apply_gradients(grads_vars)
                epoch_losses.append(loss_q.numpy())

        if (epoch + 1) % 5 == 0 and epoch_losses:
            print(
                f"[MAML-CNN] Epoch {epoch + 1}/{meta_epochs} "
                f"meta_loss={np.mean(epoch_losses):.4f}"
            )

    return maml_model


def predict_maml_cnn(maml_model, X_train, y_train, X_test):
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

    train_embeds = maml_model(X_train_tf, training=False).numpy()
    test_embeds = maml_model(X_test_tf, training=False).numpy()

    classes = np.unique(y_train)
    prototypes = []
    for c in classes:
        idx_c = np.where(y_train == c)[0]
        proto = train_embeds[idx_c].mean(axis=0)
        prototypes.append(proto)
    prototypes = np.stack(prototypes, axis=0)

    preds = []
    for e in test_embeds:
        dists = np.sum((prototypes - e) ** 2, axis=1)
        c = classes[np.argmin(dists)]
        preds.append(c)
    return np.array(preds)




print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

metric_names = ["D-index", "Accuracy", "TPR", "TNR", "PPV", "NPV"]
results_meta = {}
rows_for_csv = []

for trail in trails:
    print("\n==============================================")
    print(f" Trail {trail} ")
    print("==============================================")

    train_data = train_datasets[trail]
    test_data = test_datasets[trail]

    X_train_raw = train_data.drop(columns=['file_label']).values
    y_train_raw = train_data['file_label'].values
    X_test_raw = test_data.drop(columns=['file_label']).values
    y_test_raw = test_data['file_label'].values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    num_classes = len(np.unique(y_train))

    X_train_seq = X_train_raw.reshape(-1, X_train_raw.shape[1], 1)
    X_test_seq = X_test_raw.reshape(-1, X_test_raw.shape[1], 1)

    results_meta[trail] = {}

    print("\nTraining ProtoNet + CNN ...")
    proto_cnn_backbone = train_protonet_cnn(
        X_train=X_train_seq,
        y_train=y_train,
        embedding_dim=32,
        n_way=5,
        k_shot=5,
        q_query=5,
        meta_epochs=20,
        episodes_per_epoch=20,
        learning_rate=1e-3,
    )

    y_pred_proto_cnn = predict_protonet_cnn(
        backbone=proto_cnn_backbone,
        X_train=X_train_seq,
        y_train=y_train,
        X_test=X_test_seq,
    )

    metrics_proto_cnn = computeMeasure(num_classes, y_pred_proto_cnn, y_test)
    results_meta[trail]['ProtoNet_CNN'] = metrics_proto_cnn

    print("ProtoNet_CNN:")
    print(" [ " + ", ".join(f"{float(v):.6f}" for v in metrics_proto_cnn) + " ]")
    for name, val in zip(metric_names, metrics_proto_cnn):
        print(f" {name}: {float(val):.6f}")

    rows_for_csv.append({
        "trail": trail,
        "model": "ProtoNet_CNN",
        "D-index": float(metrics_proto_cnn[0]),
        "Accuracy": float(metrics_proto_cnn[1]),
        "TPR": float(metrics_proto_cnn[2]),
        "TNR": float(metrics_proto_cnn[3]),
        "PPV": float(metrics_proto_cnn[4]),
        "NPV": float(metrics_proto_cnn[5]),
    })

    print("\nTraining MAML + CNN ...")
    maml_cnn_model = train_maml_cnn(
        X_train=X_train_seq,
        y_train=y_train,
        num_classes=num_classes,
        meta_epochs=20,
        tasks_per_epoch=20,
        n_way=5,
        k_shot=5,
        q_query=5,
        inner_lr=5e-3,
        outer_lr=1e-3,
        inner_steps=3,
    )

    y_pred_maml_cnn = predict_maml_cnn(
        maml_cnn_model, X_train_seq, y_train, X_test_seq
    )
    metrics_maml_cnn = computeMeasure(num_classes, y_pred_maml_cnn, y_test)
    results_meta[trail]['MAML_CNN'] = metrics_maml_cnn

    print("MAML_CNN:")
    print(" [ " + ", ".join(f"{float(v):.6f}" for v in metrics_maml_cnn) + " ]")
    for name, val in zip(metric_names, metrics_maml_cnn):
        print(f" {name}: {float(val):.6f}")

    rows_for_csv.append({
        "trail": trail,
        "model": "MAML_CNN",
        "D-index": float(metrics_maml_cnn[0]),
        "Accuracy": float(metrics_maml_cnn[1]),
        "TPR": float(metrics_maml_cnn[2]),
        "TNR": float(metrics_maml_cnn[3]),
        "PPV": float(metrics_maml_cnn[4]),
        "NPV": float(metrics_maml_cnn[5]),
    })

print("\n==============================================")
print(" Meta baselines average over trails ")
print("==============================================")

agg = {}
for trail, d in results_meta.items():
    print(f"\nTrail {trail}:")
    for name, m in d.items():
        print(f" {name}: [ " + ", ".join(f"{float(v):.6f}" for v in m) + " ]")
        for metric_name, val in zip(metric_names, m):
            print(f"  {metric_name}: {float(val):.6f}")
        agg.setdefault(name, []).append(np.array(m, dtype=float))

print("\nAverage metrics over all trails:")
for name, ms in agg.items():
    avg = np.mean(ms, axis=0)
    print(f"{name}:")
    print(" [ " + ", ".join(f"{float(v):.6f}" for v in avg) + " ]")
    for metric_name, val in zip(metric_names, avg):
        print(f" {metric_name}: {float(val):.6f}")
    rows_for_csv.append({
        "trail": "avg",
        "model": name,
        "D-index": float(avg[0]),
        "Accuracy": float(avg[1]),
        "TPR": float(avg[2]),
        "TNR": float(avg[3]),
        "PPV": float(avg[4]),
        "NPV": float(avg[5]),
    })

df_results = pd.DataFrame(rows_for_csv, columns=["trail", "model"] + metric_names)
csv_path = os.path.join(curr_path, "covid_meta_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nSaved results to: {csv_path}")

