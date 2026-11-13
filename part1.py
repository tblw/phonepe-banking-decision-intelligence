import os
import numpy as np

def run_part1():
    os.makedirs("outputs", exist_ok=True)
    
    # Load data
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    # Columns: record_id(0), X1(1), X2(2), X3(3), X4(4), X5(5), X6(6), X7(7), outcome(8), value(9)
    x1 = data[:, 1]
    x2 = data[:, 2]
    x3 = data[:, 3]
    x4 = data[:, 4]
    x5 = data[:, 5]
    x6 = data[:, 6]
    x7 = data[:, 7]
    y = data[:, 9]
    n = len(y)

    # 1.1 Estimation
    mean_value = np.mean(y)
    sd_value = np.std(y, ddof=1)
    se_value = sd_value / np.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # 1.2 Test H0: mu = 8.0
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # 1.3 Sample size
    E = 0.5
    n_required = int(np.ceil((1.96 * sd_value / E) ** 2))

    # 1.4 Simple regression (y on X4)
    x4_mean = np.mean(x4)
    y_mean = mean_value
    cov_x4_y = np.cov(x4, y, ddof=1)[0, 1]
    var_x4 = np.var(x4, ddof=1)
    reg_slope = cov_x4_y / var_x4
    reg_intercept = y_mean - reg_slope * x4_mean
    fitted_14 = reg_slope * x4 + reg_intercept
    
    tss = np.sum((y - y_mean) ** 2)
    ess = np.sum((fitted_14 - y_mean) ** 2)
    rss = np.sum((y - fitted_14) ** 2)
    tss_equals_ess_plus_rss = "yes" if abs(tss - (ess + rss)) < 1e-6 * tss else "no"
    r2 = 1.0 - (rss / tss)
    rmse = np.sqrt(rss / n)
    mae = np.mean(np.abs(y - fitted_14))

    # 1.5 Gradient descent on standardised X4
    x4_std = np.std(x4, ddof=1)
    z_feat = (x4 - x4_mean) / x4_std
    m, c = 0.0, 0.0
    eta = 0.1
    
    def get_cost(m_val, c_val):
        preds = m_val * z_feat + c_val
        return np.mean((y - preds) ** 2)

    gd_J0 = get_cost(m, c)
    costs = {}
    for epoch in range(1, 501):
        preds = m * z_feat + c
        err = y - preds
        dm = -(2.0 / n) * np.sum(z_feat * err)
        dc = -(2.0 / n) * np.sum(err)
        m -= eta * dm
        c -= eta * dc
        if epoch in [1, 10, 100, 500]:
            costs[epoch] = get_cost(m, c)

    gd_J1 = costs[1]
    gd_J10 = costs[10]
    gd_J100 = costs[100]
    gd_J500 = costs[500]

    gd_slope = m / x4_std
    gd_intercept = c - m * x4_mean / x4_std
    gd_slope_gap = abs(gd_slope - reg_slope)

    # 1.6 Funnel check and log remedy
    # Stable sort by fitted values
    sort_idx = np.argsort(fitted_14, kind='stable')
    y_sorted = y[sort_idx]
    fitted_sorted = fitted_14[sort_idx]
    lower_resids = y_sorted[:400] - fitted_sorted[:400]
    upper_resids = y_sorted[400:] - fitted_sorted[400:]
    
    resid_sd_lower_raw = np.std(lower_resids, ddof=1)
    resid_sd_upper_raw = np.std(upper_resids, ddof=1)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log model
    log_y = np.log(y)
    log_x4 = np.log(x4)
    lx4_mean = np.mean(log_x4)
    ly_mean = np.mean(log_y)
    loglog_slope = np.cov(log_x4, log_y, ddof=1)[0, 1] / np.var(log_x4, ddof=1)
    loglog_intercept = ly_mean - loglog_slope * lx4_mean
    fitted_log = loglog_slope * log_x4 + loglog_intercept
    log_rss = np.sum((log_y - fitted_log) ** 2)
    log_tss = np.sum((log_y - ly_mean) ** 2)
    loglog_r2 = 1.0 - (log_rss / log_tss)
    loglog_rmse = np.sqrt(log_rss / n)

    sort_idx_log = np.argsort(fitted_log, kind='stable')
    log_y_sorted = log_y[sort_idx_log]
    fitted_log_sorted = fitted_log[sort_idx_log]
    lower_resids_log = log_y_sorted[:400] - fitted_log_sorted[:400]
    upper_resids_log = log_y_sorted[400:] - fitted_log_sorted[400:]
    
    resid_sd_lower_log = np.std(lower_resids_log, ddof=1)
    resid_sd_upper_log = np.std(upper_resids_log, ddof=1)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # 1.7 Multiple regression on log variables
    # X columns: const, ln(X4), X1, X2, X3, X5, X6, X7
    X_mat = np.column_stack([
        np.ones(n),
        log_x4,
        x1, x2, x3, x5, x6, x7
    ])
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ log_y)
    mr_beta_0, mr_beta_1, mr_beta_2, mr_beta_3, mr_beta_4, mr_beta_5, mr_beta_6, mr_beta_7 = beta
    
    mr_fitted = X_mat @ beta
    mr_rss = np.sum((log_y - mr_fitted) ** 2)
    mr_r2 = 1.0 - (mr_rss / log_tss)
    p_mult = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mult - 1)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    # Write outputs/part1_results.txt
    results = [
        f"n = {n}",
        f"mean_value = {mean_value:.4f}",
        f"sd_value = {sd_value:.4f}",
        f"se_value = {se_value:.4f}",
        f"ci95_low = {ci95_low:.4f}",
        f"ci95_high = {ci95_high:.4f}",
        f"z_stat = {z_stat:.4f}",
        f"reject_h0 = {reject_h0}",
        f"n_required = {n_required}",
        f"reg_slope = {reg_slope:.4f}",
        f"reg_intercept = {reg_intercept:.4f}",
        f"tss = {tss:.4f}",
        f"ess = {ess:.4f}",
        f"rss = {rss:.4f}",
        f"tss_equals_ess_plus_rss = {tss_equals_ess_plus_rss}",
        f"r2 = {r2:.4f}",
        f"rmse = {rmse:.4f}",
        f"mae = {mae:.4f}",
        f"gd_J0 = {gd_J0:.4f}",
        f"gd_J1 = {gd_J1:.4f}",
        f"gd_J10 = {gd_J10:.4f}",
        f"gd_J100 = {gd_J100:.4f}",
        f"gd_J500 = {gd_J500:.4f}",
        f"gd_slope = {gd_slope:.4f}",
        f"gd_intercept = {gd_intercept:.4f}",
        f"gd_slope_gap = {gd_slope_gap:.4f}",
        f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}",
        f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}",
        f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}",
        f"loglog_intercept = {loglog_intercept:.4f}",
        f"loglog_slope = {loglog_slope:.4f}",
        f"loglog_r2 = {loglog_r2:.4f}",
        f"loglog_rmse = {loglog_rmse:.4f}",
        f"resid_sd_lower_log = {resid_sd_lower_log:.4f}",
        f"resid_sd_upper_log = {resid_sd_upper_log:.4f}",
        f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}",
        f"mr_beta_0 = {mr_beta_0:.4f}",
        f"mr_beta_1 = {mr_beta_1:.4f}",
        f"mr_beta_2 = {mr_beta_2:.4f}",
        f"mr_beta_3 = {mr_beta_3:.4f}",
        f"mr_beta_4 = {mr_beta_4:.4f}",
        f"mr_beta_5 = {mr_beta_5:.4f}",
        f"mr_beta_6 = {mr_beta_6:.4f}",
        f"mr_beta_7 = {mr_beta_7:.4f}",
        f"mr_r2 = {mr_r2:.4f}",
        f"mr_adj_r2 = {mr_adj_r2:.4f}",
        f"loglog_adj_r2 = {loglog_adj_r2:.4f}"
    ]

    with open("outputs/part1_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part1()