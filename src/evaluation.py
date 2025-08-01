# src/evaluation.py
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np
import shap
import os
plots_dir = r"C:\Users\Mahen\PycharmProjects\pythonProject20\loan default prediction\plots"
os.makedirs(plots_dir, exist_ok=True)


def optimize_threshold(y_true, y_proba):
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresh[best_idx] if len(thresh) > 0 else 0.5
    print(f"🎯 Best threshold for F1: {best_thresh:.4f}")
    return best_thresh


def evaluate_model(y_test, y_proba, X_test_flat, model, save_plots=True):

    print("📊 Evaluating model...")

    # Optimize threshold
    best_thresh = optimize_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)

    # Metrics
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fully Paid', 'Charged Off']))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)
    print(f"🔤 PR-AUC Score: {pr_auc:.4f}")

    # Plot ROC & PR
    _plot_evaluation_curves(y_test, y_proba, roc_auc, pr_auc, save_plots)

    # SHAP Explainability
    print("🧠 Generating SHAP explanations...")
    _generate_shap_plots(model, X_test_flat, save_plots)

    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'best_threshold': best_thresh,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def _plot_evaluation_curves(y_test, y_proba, roc_auc, pr_auc, save_plots=True):
    """Plot ROC and Precision-Recall curves and save to plots folder."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax[0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
    ax[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].set_title("ROC Curve"); ax[0].legend(); ax[0].grid(True)

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ax[1].plot(rec, prec, label=f'PR-AUC = {pr_auc:.4f}')
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve"); ax[1].legend(); ax[1].grid(True)

    plt.tight_layout()
    if save_plots:
        path = os.path.join(plots_dir, "evaluation_roc_pr.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ ROC & PR curves saved to: {path}")
    plt.show()


def _generate_shap_plots(model, X_test_flat, save_plots=True):
    """Generate SHAP summary and force plots and save to plots folder."""
    try:
        base_model = model.named_estimators_['xgb']
    except:
        print("⚠️ Could not access base estimator for SHAP.")
        return

    # Sample for speed
    X_sample = X_test_flat.sample(n=min(100, len(X_test_flat)), random_state=42)

    # Explain
    explainer = shap.TreeExplainer(base_model, feature_names=X_sample.columns.tolist())
    shap_values = explainer.shap_values(X_sample)

    # === SHAP Summary Plot ===
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Impact on Prediction)")
    plt.tight_layout()
    if save_plots:
        path = os.path.join(plots_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ SHAP summary plot saved to: {path}")
    plt.show()

    # === SHAP Force Plot (for first instance) ===
    shap.initjs()
    try:
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_sample.iloc[0],
            matplotlib=True,
            show=False
        )
        # Save as image not possible directly with force_plot, so just show
        # For full export, use JavaScript version in Jupyter or web app
        print("💡 SHAP force plot shown above (interactive).")
    except Exception as e:
        print(f"⚠️ Could not generate force plot: {e}")

def show_prediction_explanation(model, loan_data, feature_names):
    """
    Explain a single loan prediction.
    loan_data: 1-row DataFrame with same features
    """
    base_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(base_model, feature_names=feature_names)
    shap_value = explainer.shap_values(loan_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_value, loan_data)