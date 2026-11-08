"""Small reproducible mathematical checks, NOT a real-data training experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import sys
import torch
from query_mixup import query_conditioned_mixup


def check() -> dict:
    dtype = torch.float64
    images = torch.eye(2, dtype=dtype)
    texts = torch.tensor([[2., 0., 1.], [0., 3., -1.]], dtype=dtype)
    labels = torch.eye(2, dtype=dtype)
    # Both positive endpoint pairs vs both negative endpoint pairs at l=1/2.
    pos_input = (.5 * images[0] + .5 * images[1],
                 .5 * texts[0] + .5 * texts[1])
    neg_input = (.5 * images[0] + .5 * images[1],
                 .5 * texts[1] + .5 * texts[0])
    collision = dict(identical_images=bool(torch.equal(pos_input[0], neg_input[0])),
                     identical_texts=bool(torch.equal(pos_input[1], neg_input[1])),
                     legacy_target_from_positive_pairs=1.,
                     legacy_target_from_negative_pairs=0.,
                     exact_mix_coefficient=.5,
                     caveat='Continuous Beta sampling has zero probability of exactly 0.5. '
                            'This proves a possible endpoint-supervision inconsistency, not its '
                            'frequency or its causal impact on NUS/INRIA performance.')
    # Anchor the same cat image to a .5 cat-text/.5 dog-text mixture.
    # Generate it from either candidate order; both definitions agree.
    i = torch.tensor([0, 0]); j = torch.tensor([0, 1])
    p = torch.tensor([1, 0]); lam = torch.full((2, 1), .5, dtype=dtype)
    img, txt, target = query_conditioned_mixup(images, texts, labels, i, j,
        permutation=p, coefficients=lam, image_is_query=torch.ones(2, dtype=torch.bool))
    anchored = dict(identical_queries=bool(torch.equal(img[0], img[1])),
                    identical_mixed_candidates=bool(torch.equal(txt[0], txt[1])),
                    targets=target.flatten().tolist())
    # Keeping Beta mean at .5 while increasing strength keeps reducing expected CE.
    concentration = []
    for s in (4., 20., 200., 2000.):
        strength = torch.tensor(s, dtype=dtype)
        a = b = strength / 2
        loss = torch.digamma(strength) - .5 * torch.digamma(a) - .5 * torch.digamma(b)
        au = torch.digamma(strength+1) - .5*torch.digamma(a+1) - .5*torch.digamma(b+1)
        concentration.append(dict(strength=s,alpha=float(a),beta=float(b),mean=.5,
                                  soft_target=.5,expected_ce=float(loss),
                                  model_AU=float(au),model_EU=math.log(2)-float(au)))
    return dict(environment=dict(python=sys.version.split()[0],torch=str(torch.__version__),device='cpu'),
                source_commit='6fc7fd56baef92ce43f1a32a52c1fbc100035c80',
                legacy_pair_mixup_collision=collision,query_conditioned_check=anchored,
                concentration_check=concentration,
                interpretation='A soft mean target alone does not identify epistemic concentration; '
                               'no counterexample here proves real-data performance degradation or '
                               'invalidates VLS empirical results.',
                new_INRIA_or_NUS_training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(check(), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError('Do not overwrite: '+str(args.output))
        args.output.write_text(result+'\n', encoding='utf8')
    print(result)
