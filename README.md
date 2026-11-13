Domain: Banking

# PhonePe Banking Decision Intelligence - Capstone Project

## Part 1 findings
- **R1.1**: H0 was rejected (z_stat = 10.4281, critical value = 1.96). A Type I error here means falsely concluding that the true average monthly spend differs from INR 8,000 when it actually equals INR 8,000. The gap between sample mean (8.9382) and the planning target (8.0) is about INR 938, which matters in practice for budget allocation.
- **R1.2**: n_required is 124, which is well below the 800 available rows, confirming ample statistical power for the team's monthly spend estimations.
- **R1.3**: gd_slope_gap is 0.0000, showing that batch gradient descent successfully converged to the closed-form ordinary least squares solution.
- **R1.4**: Resid sd ratio raw was 1.8542 indicating heteroskedasticity (funnel shape), which the log-log model reduced successfully to 1.0215.
- **R1.5**: Predictors with absolute coefficients above 0.05 in the multiple regression are ln(X4), X1, X2, and X3. Comparing adjusted R² shows that adding multivariate predictors improves explanatory power significantly over the univariate log model.

## Part 2 findings
- **R2.1**: Learning rate comparison shows that 0.01 underfits (too small), 0.1 converges steadily, and 1.0 diverges or oscillates wildly.
- **R2.2**: The top predictors by absolute weight are credit utilization ratio, account tenure, and failed transaction count.
- **R2.3**: Blind accuracy is 0.7875 compared to 0.8125 at tau = 0.5. Lowering tau from 0.5 to 0.3 increased recall while trading off precision.
- **R2.4**: KNN and Logistic regression show comparable performance on F1 and recall metrics on test data.
- **R2.5**: The deployment threshold chosen via cost matrix minimization accounts for the asymmetric penalty of missed defaults versus false alarms. Thresholds were optimized on training data to avoid data leakage.

## Part 3 findings
- **R3.1**: WCSS drops across K=2 to K=6 support K=3 as the optimal elbow point for segmenting borrower behavior.
- **R3.2**: Segments profiled into Prime Borrowers, Active Revolvers, and High-Risk Borrowers based on inflows, credit utilization, and default rates.
- **R3.3**: Hierarchical clustering cut analysis reveals distinct structural groupings among borrower cohorts across single, complete, and average linkage methods.
- **R3.4 & R3.5**: PCA confirms that the first two principal components capture over 80% of total variance across standardized continuous features.