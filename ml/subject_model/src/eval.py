"""Metrics for finalized float models; no fitting or checkpoint selection."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import torch


def evaluate(model, split, batch_size=64):
    model.eval()
    total_loss, predictions = 0.0, []
    with torch.inference_mode():
        for start in range(0, len(split.y), batch_size):
            x, y = split.x[start:start + batch_size], split.y[start:start + batch_size]
            logits = model(x)
            total_loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
            predictions.extend(logits.argmax(1).tolist())
    y = split.y.numpy()
    predicted = np.asarray(predictions)
    return {
        'n': len(y), 'loss': total_loss / len(y),
        'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        'confusion_matrix_order': ['left_hand', 'right_hand'],
        'confusion_matrix_axes': 'rows=true, columns=predicted',
    }


def main():
    import argparse
    import json
    from pathlib import Path
    from src.data.preprocess import Split
    from src.models.tinyeegnet import TinyEEGNet
    parser = argparse.ArgumentParser(description='Evaluate a finalized saved combined model')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=['demonstration', 'test'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.run / 'best_checkpoint.pt', weights_only=True)
    model = TinyEEGNet(dropout=checkpoint['config']['model']['dropout'])
    model.load_state_dict(checkpoint['model_state'])
    prepared = torch.load(args.run / 'prepared_tensors.pt', weights_only=True)[args.split]
    print(json.dumps(evaluate(model, Split(prepared['x'], prepared['y'], ())), indent=2))


if __name__ == '__main__':
    main()
