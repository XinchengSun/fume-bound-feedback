"""Evaluate/replay a v5 checkpoint, using its exact labels, W, evidence heads and prior."""
import argparse
from pathlib import Path
import torch
if __package__:
    from .data import load_data, sha256_file
    from .model import MultiLabelV5, ModelConfig
    from .metrics import evaluate, save_json
    from .train import FORMAT
else:
    from data import load_data, sha256_file
    from model import MultiLabelV5, ModelConfig
    from metrics import evaluate, save_json
    from train import FORMAT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    if min(a.batch_size, a.chunk, a.threads) < 1:
        raise ValueError('Positive sizes/threads required')
    torch.set_num_threads(a.threads)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError('Requires v5 checkpoint; old pair-head and encoder-only files are incompatible')
    datasets, receipt = load_data(a.data, saved['model_config']['classes'], saved['run_config']['label_base'])
    if receipt['file_sha256'] != saved['data_receipt']['file_sha256']:
        raise ValueError('Data SHA mismatch: this replay entrypoint does not silently switch datasets')
    model = MultiLabelV5(ModelConfig(**saved['model_config'])).to(a.device)
    model.load_state_dict(saved['model_state'], strict=True)
    report = evaluate(model, datasets[a.split], a.output, a.batch_size, a.chunk)
    save_json(a.output/'replay.json', dict(split=a.split, epoch=saved['epoch'],
        checkpoint_sha256=sha256_file(a.checkpoint), data_sha256=receipt['file_sha256']))
    print(report['retrieval'])


if __name__ == '__main__':
    main()
