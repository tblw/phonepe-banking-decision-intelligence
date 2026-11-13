import os
import numpy as np

def run_part2():
    os.makedirs("outputs", exist_ok=True)
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    features = data[:, 1:8] # X1 to X7
    outcome = data[:, 8]
    n = len(outcome)

    # 2.1 Stratified 70/30 split
    idx_0 = np.where(outcome == 0)[0]
    idx_1 = np.where(outcome == 1)[0]
    
    n0_train = int(7 * len(idx_0) / 10)
    n1_train = int(7 * len(idx_1) / 10)

    train_idx = np.concatenate([idx_0[:n0_train], idx_1[:n1_train]])
    test_idx = np.concatenate([idx_0[n0_train:], idx_1[n1_train:]])

    X_train_raw = features[train_idx]
    y_train = outcome[train_idx]
    X_test_raw = features[test_idx]
    y_test = outcome[test_idx]

    train_rows = len(train_idx)
    train_positives = int(np.sum(y_train))
    test_rows = len(test_idx)
    test_positives = int(np.sum(y_test))

    # Standardise using train mean and sample SD
    mu_train = np.mean(X_train_raw, axis=0)
    sd_train = np.std(X_train_raw, axis=0, ddof=1)
    
    X_train = (X_train_raw - mu_train) / sd_train
    X_test = (X_test_raw - mu_train) / sd_train

    # 2.2 Logistic regression from scratch
    ntrain = len(y_train)
    weights = np.zeros(7)
    bias = 0.0
    eta = 0.1

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def bce_loss(y_true, p_pred):
        eps = 1e-15
        p_pred = np.clip(p_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))

    loss_epoch_0 = bce_loss(y_train, sigmoid(X_train @ weights + bias))

    losses = {}
    for epoch in range(1, 1001):
        z = X_train @ weights + bias
        preds = sigmoid(z)
        err = preds - y_train
        dw = (1.0 / ntrain) * (X_train.T @ err)
        db = (1.0 / ntrain) * np.sum(err)
        weights -= eta * dw
        bias -= eta * db
        if epoch in [1, 10, 100, 1000]:
            losses[epoch] = bce_loss(y_train, sigmoid(X_train @ weights + bias))

    loss_epoch_1 = losses[1]
    loss_epoch_10 = losses[10]
    loss_epoch_100 = losses[100]
    loss_epoch_1000 = losses[1000]

    test_preds = sigmoid(X_test @ weights + bias)
    test_bce = bce_loss(y_test, test_preds)

    # 2.3 Learning rate effect (200 updates)
    def train_lr(lr):
        w = np.zeros(7)
        b = 0.0
        for _ in range(200):
            preds = sigmoid(X_train @ w + b)
            err = preds - y_train
            dw = (1.0 / ntrain) * (X_train.T @ err)
            db = (1.0 / ntrain) * np.sum(err)
            w -= lr * dw
            b -= lr * db
        return bce_loss(y_train, sigmoid(X_train @ w + b))

    loss_epoch_200_eta_0_01 = train_lr(0.01)
    loss_epoch_200_eta_0_1 = train_lr(0.1)
    loss_epoch_200_eta_1_0 = train_lr(1.0)

    # 2.4 Odds ratios
    odds_ratios = np.exp(weights)
    odds_ratio_X1, odds_ratio_X2, odds_ratio_X3, odds_ratio_X4, odds_ratio_X5, odds_ratio_X6, odds_ratio_X7 = odds_ratios

    # 2.5 Evaluate at tau = 0.5 and 0.3
    def evaluate_metrics(y_true, probs, tau):
        preds = (probs >= tau).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return tp, fp, tn, fn, acc, prec, rec, f1, spec

    tp_05, fp_05, tn_05, fn_05, acc_05, prec_05, rec_05, f1_05, spec_05 = evaluate_metrics(y_test, test_preds, 0.5)
    tp_03, fp_03, tn_03, fn_03, acc_03, prec_03, rec_03, f1_03, spec_03 = evaluate_metrics(y_test, test_preds, 0.3)
    blind_accuracy = np.sum(y_test == 0) / len(y_test)

    # 2.6 Threshold sweep CSV
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_rows = []
    for t in taus:
        tp, fp, tn, fn, _, _, rec, _, spec = evaluate_metrics(y_test, test_preds, t)
        tpr = rec
        fpr = 1.0 - spec
        sweep_rows.append(f"{t:.1f},{tp},{fp},{tn},{fn},{tpr:.4f},{fpr:.4f}")

    with open("outputs/part2_thresholds.csv", "w") as f:
        f.write("tau,tp,fp,tn,fn,tpr,fpr\n" + "\n".join(sweep_rows) + "\n")

    # 2.7 KNN with 5-fold CV
    k_vals = list(range(1, 22, 2))
    cv_rows = []
    best_k_f1 = -1.0
    knn_best_k = 1

    for k in k_vals:
        fold_f1s = []
        for fold in range(5):
            val_indices = np.where(np.arange(ntrain) % 5 == fold)[0]
            train_indices = np.where(np.arange(ntrain) % 5 != fold)[0]
            
            ref_X = X_train[train_indices]
            ref_y = y_train[train_indices]
            val_X = X_train[val_indices]
            val_y = y_train[val_indices]

            preds_val = []
            for q in val_X:
                dists = np.sqrt(np.sum((ref_X - q) ** 2, axis=1))
                sort_idx = np.argsort(dists, kind='stable')
                top_k_indices = sort_idx[:k]
                top_k_labels = ref_y[top_k_indices]
                pos_count = np.sum(top_k_labels == 1)
                pred = 1 if pos_count >= (k + 1) / 2 else 0
                preds_val.append(pred)
            
            preds_val = np.array(preds_val)
            tp = np.sum((preds_val == 1) & (val_y == 1))
            fp = np.sum((preds_val == 1) & (val_y == 0))
            fn = np.sum((preds_val == 0) & (val_y == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            fold_f1s.append(f1)
        
        mean_cv_f1 = np.mean(fold_f1s)
        cv_rows.append(f"{k},{mean_cv_f1:.4f}")
        if mean_cv_f1 > best_k_f1:
            best_k_f1 = mean_cv_f1
            knn_best_k = k

    with open("outputs/part2_knn_cv.csv", "w") as f:
        f.write("K,mean_cv_f1\n" + "\n".join(cv_rows) + "\n")

    # Refit KNN with all 560 training rows
    knn_preds = []
    for q in X_test:
        dists = np.sqrt(np.sum((X_train - q) ** 2, axis=1))
        sort_idx = np.argsort(dists, kind='stable')
        top_k_indices = sort_idx[:knn_best_k]
        pos_count = np.sum(y_train[top_k_indices] == 1)
        pred = 1 if pos_count >= (knn_best_k + 1) / 2 else 0
        knn_preds.append(pred)
    
    knn_preds = np.array(knn_preds)
    knn_tp = int(np.sum((knn_preds == 1) & (y_test == 1)))
    knn_fp = int(np.sum((knn_preds == 1) & (y_test == 0)))
    knn_tn = int(np.sum((knn_preds == 0) & (y_test == 0)))
    knn_fn = int(np.sum((knn_preds == 0) & (y_test == 1)))
    knn_accuracy = (knn_tp + knn_tn) / len(y_test)
    knn_precision = knn_tp / (knn_tp + knn_fp) if (knn_tp + knn_fp) > 0 else 0.0
    knn_recall = knn_tp / (knn_tp + knn_fn) if (knn_tp + knn_fn) > 0 else 0.0
    knn_f1 = (2 * knn_precision * knn_recall) / (knn_precision + knn_recall) if (knn_precision + knn_recall) > 0 else 0.0

    # 2.8 Money cost threshold selection
    train_probs = sigmoid(X_train @ weights + bias)
    best_cost = float('inf')
    cost_tau = 0.05
    for k_t in range(1, 20):
        t = k_t / 20.0
        tp, fp, tn, fn, _, _, _, _, _ = evaluate_metrics(y_train, train_probs, t)
        cost = 50 * fp + 400 * fn
        if cost < best_cost:
            best_cost = cost
            cost_tau = t

    cost_train_at_tau = best_cost
    
    tp_ct, fp_ct, tn_ct, fn_ct, _, _, _, _, _ = evaluate_metrics(y_test, test_preds, cost_tau)
    cost_test_at_tau = 50 * fp_ct + 400 * fn_ct

    tp_05t, fp_05t, tn_05t, fn_05t, _, _, _, _, _ = evaluate_metrics(y_test, test_preds, 0.5)
    cost_test_at_0_5 = 50 * fp_05t + 400 * fn_05t

    cost_test_flag_nobody = 400 * test_positives

    results = [
        f"train_rows = {train_rows}",
        f"train_positives = {train_positives}",
        f"test_rows = {test_rows}",
        f"test_positives = {test_positives}",
        f"loss_epoch_0 = {loss_epoch_0:.4f}",
        f"loss_epoch_1 = {loss_epoch_1:.4f}",
        f"loss_epoch_10 = {loss_epoch_10:.4f}",
        f"loss_epoch_100 = {loss_epoch_100:.4f}",
        f"loss_epoch_1000 = {loss_epoch_1000:.4f}",
        f"weight_X1 = {weights[0]:.4f}",
        f"weight_X2 = {weights[1]:.4f}",
        f"weight_X3 = {weights[2]:.4f}",
        f"weight_X4 = {weights[3]:.4f}",
        f"weight_X5 = {weights[4]:.4f}",
        f"weight_X6 = {weights[5]:.4f}",
        f"weight_X7 = {weights[6]:.4f}",
        f"bias = {bias:.4f}",
        f"test_bce = {test_bce:.4f}",
        f"loss_epoch_200_eta_0.01 = {loss_epoch_200_eta_0_01:.4f}",
        f"loss_epoch_200_eta_0.1 = {loss_epoch_200_eta_0_1:.4f}",
        f"loss_epoch_200_eta_1.0 = {loss_epoch_200_eta_1_0:.4f}",
        f"odds_ratio_X1 = {odds_ratio_X1:.4f}",
        f"odds_ratio_X2 = {odds_ratio_X2:.4f}",
        f"odds_ratio_X3 = {odds_ratio_X3:.4f}",
        f"odds_ratio_X4 = {odds_ratio_X4:.4f}",
        f"odds_ratio_X5 = {odds_ratio_X5:.4f}",
        f"odds_ratio_X6 = {odds_ratio_X6:.4f}",
        f"odds_ratio_X7 = {odds_ratio_X7:.4f}",
        f"tau_0.5_tp = {tp_05}",
        f"tau_0.5_fp = {fp_05}",
        f"tau_0.5_tn = {tn_05}",
        f"tau_0.5_fn = {fn_05}",
        f"tau_0.5_accuracy = {acc_05:.4f}",
        f"tau_0.5_precision = {prec_05:.4f}",
        f"tau_0.5_recall = {rec_05:.4f}",
        f"tau_0.5_f1 = {f1_05:.4f}",
        f"tau_0.5_specificity = {spec_05:.4f}",
        f"tau_0.3_tp = {tp_03}",
        f"tau_0.3_fp = {fp_03}",
        f"tau_0.3_tn = {tn_03}",
        f"tau_0.3_fn = {fn_03}",
        f"tau_0.3_accuracy = {acc_03:.4f}",
        f"tau_0.3_precision = {prec_03:.4f}",
        f"tau_0.3_recall = {rec_03:.4f}",
        f"tau_0.3_f1 = {f1_03:.4f}",
        f"tau_0.3_specificity = {spec_03:.4f}",
        f"blind_accuracy = {blind_accuracy:.4f}",
        f"knn_best_k = {knn_best_k}",
        f"knn_tp = {knn_tp}",
        f"knn_fp = {knn_fp}",
        f"knn_tn = {knn_tn}",
        f"knn_fn = {knn_fn}",
        f"knn_accuracy = {knn_accuracy:.4f}",
        f"knn_precision = {knn_precision:.4f}",
        f"knn_recall = {knn_recall:.4f}",
        f"knn_f1 = {knn_f1:.4f}",
        f"cost_tau = {cost_tau:.2f}",
        f"cost_train_at_tau = {cost_train_at_tau}",
        f"cost_test_at_tau = {cost_test_at_tau}",
        f"cost_test_at_0.5 = {cost_test_at_0_5}",
        f"cost_test_flag_nobody = {cost_test_flag_nobody}"
    ]

    with open("outputs/part2_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part2()