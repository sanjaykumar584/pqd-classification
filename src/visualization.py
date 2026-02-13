"""
Visualization helpers for Power Quality Disturbance Classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_signal_gallery(signals, labels, class_names, n_examples=3,
                        time_ms=None, figsize=None):
    """Plot a gallery of waveforms, one subplot per class.

    Parameters
    ----------
    signals : np.ndarray, shape (n_samples, signal_length)
    labels : np.ndarray, shape (n_samples,)
    class_names : list of str
    n_examples : int
        Number of random signals to overlay per class.
    time_ms : np.ndarray or None
        Time axis in milliseconds. If None, uses sample indices.
    figsize : tuple or None
    """
    n_classes = len(class_names)
    ncols = 3
    nrows = int(np.ceil(n_classes / ncols))
    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, cls in enumerate(class_names):
        ax = axes[idx]
        mask = labels == cls
        cls_signals = signals[mask]
        n_avail = min(n_examples, len(cls_signals))
        chosen = cls_signals[np.random.choice(len(cls_signals), n_avail, replace=False)]

        x = time_ms if time_ms is not None else np.arange(signals.shape[1])
        for s in chosen:
            ax.plot(x, s, alpha=0.7, linewidth=0.8)

        ax.set_title(cls.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_ylim(-1.5, 1.5)
        if time_ms is not None:
            ax.set_xlabel('Time (ms)', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_class_distribution(labels, title='Class Distribution', figsize=(12, 5)):
    """Bar chart of samples per class."""
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(unique)), counts[order], color=sns.color_palette('viridis', len(unique)))
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(unique[order], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Samples')
    ax.set_title(title, fontweight='bold')

    for bar, count in zip(bars, counts[order]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix',
                          figsize=(12, 10), normalize=False):
    """Plot an annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_model_comparison(results_dict, metric='accuracy', title=None, figsize=(10, 6)):
    """Bar chart comparing models.

    Parameters
    ----------
    results_dict : dict
        {model_name: {'accuracy': float, 'f1_macro': float, ...}}
    metric : str
        Key to plot.
    title : str or None
    figsize : tuple
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette('Set2', len(models))
    bars = ax.bar(models, values, color=colors)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Model Comparison — {metric.replace("_", " ").title()}',
                 fontweight='bold')
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names, top_n=20,
                            title='Feature Importance', figsize=(10, 8)):
    """Horizontal bar chart of top feature importances."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(top_n), importances[indices], color=sns.color_palette('viridis', top_n))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_pca_2d(X_2d, labels, class_names, title='PCA Projection', figsize=(10, 8)):
    """2D scatter plot colored by class."""
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette('tab20', len(class_names))

    for i, cls in enumerate(class_names):
        mask = labels == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=15, alpha=0.6,
                   label=cls.replace('_', ' '), color=palette[i])

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=2)
    plt.tight_layout()
    return fig
