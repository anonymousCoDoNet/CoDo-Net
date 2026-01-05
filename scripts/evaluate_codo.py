"""Evaluation script for CoDo-Net using cached features.

This script evaluates a trained CoDo-Net checkpoint on test data using
pre-extracted features, computing AUC, Accuracy, Precision, Recall, and F1.

Usage:
    python scripts/evaluate_codo.py \
        --feature_dir data/cached_features/test \
        --file_list data/test_list.txt \
        --checkpoint checkpoints/codo_net/best_codo.pt \
        --output_dir results/codo \
        --threshold 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_forensics.models.codo_net import ImprovedCoDoNet
from hybrid_forensics.datasets import create_dataloader


def evaluate_from_cache(
    feature_dir: str,
    file_list: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 16,
    seq_len: int = 64,
    num_workers: int = 4,
    threshold: float = 0.5,
) -> dict:
    """Evaluate CoDo-Net using cached features.

    Args:
        feature_dir: Directory containing cached features
        file_list: Path to test file_list.txt
        checkpoint_path: Path to trained CoDo-Net checkpoint
        output_dir: Output directory for results
        batch_size: Batch size for evaluation
        seq_len: Temporal sequence length
        num_workers: Number of data loading workers
        threshold: Decision threshold for classification (default: 0.5)

    Returns:
        Dictionary containing evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("CoDo-Net Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Feature directory: {feature_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Decision threshold: {threshold}")

    # Load model
    print("\n[1/3] Loading CoDo-Net...")
    # DynamicForensics MS-TCN outputs 768-dim temporal features
    model = ImprovedCoDoNet(sem_dim=2048, dyn_dim=768).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    if 'epoch' in ckpt:
        print(f"  Checkpoint: Epoch {ckpt['epoch']}")
    if 'metrics' in ckpt and 'val_acc' in ckpt['metrics']:
        print(f"  Validation accuracy: {ckpt['metrics']['val_acc']:.4f}")

    print("✓ Model loaded successfully")

    # Load data
    print(f"\n[2/3] Loading test data...")
    _, test_loader = create_dataloader(
        file_list=file_list,
        feature_root=feature_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seq_len=seq_len,
        require_dynamic=True,
        use_concat=True,
    )
    print(f"✓ Loaded {len(test_loader.dataset)} test samples")

    # Evaluate
    print(f"\n[3/3] Evaluating...")
    all_labels = []
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            labels = batch['label'].to(device)
            sem = batch['sem_feat'].to(device)
            dyn = batch['dyn_feat'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            output = model(sem, dyn, mask=mask)
            probs = torch.softmax(output.logits, dim=-1)
            real_probs = probs[:, 1]  # Probability of Real class
            # Use custom threshold for prediction
            preds = (real_probs >= threshold).long()

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(real_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    labels = np.array(all_labels)
    scores = np.array(all_scores)
    preds = np.array(all_preds)

    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"AUC:       {auc:.4f} ({auc*100:.2f}%)")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 60)

    # Confusion matrix
    cm = metrics.confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              Fake  Real")
    print(f"Actual Fake   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Real   {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'metrics': {
            'auc': float(auc),
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        },
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
        'checkpoint': checkpoint_path,
        'num_samples': len(labels),
        'num_real': int((labels == 1).sum()),
        'num_fake': int((labels == 0).sum()),
        'predictions': {
            'labels': labels.tolist(),
            'scores': scores.tolist(),  # Real class probabilities
            'preds': preds.tolist(),
        },
    }

    output_file = os.path.join(output_dir, 'codo_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoDo-Net using cached features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--feature_dir',
        help='Directory containing cached test features (alias for --test_feature_dir)'
    )
    parser.add_argument(
        '--test_feature_dir',
        help='Directory containing cached test features (overrides --feature_dir)'
    )
    parser.add_argument(
        '--file_list',
        required=True,
        help='Path to test file_list.txt'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to trained CoDo-Net checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output_dir',
        default='results/codo',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=64,
        help='Temporal sequence length'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold for classification (default: 0.5)'
    )

    args = parser.parse_args()

    # Handle feature directory with backward compatibility
    test_feature_dir = args.test_feature_dir if args.test_feature_dir else args.feature_dir
    if not test_feature_dir:
        parser.error("Either --test_feature_dir or --feature_dir must be specified")

    evaluate_from_cache(
        feature_dir=test_feature_dir,
        file_list=args.file_list,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
