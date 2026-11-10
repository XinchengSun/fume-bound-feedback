from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"certification"))
from eligibility import classify,load_manifest,REVIEWS


def row(name, **fields):
    data=dict(record_id=name,candidate=True,modality="image",canonical_id=name,
              source_block="block-"+name,uses=[],aliases=[])
    return {**data,**dict.fromkeys(REVIEWS,"e"),**fields}


class Eligibility(unittest.TestCase):
    def test_known_use_blocks_transitive_alias_and_sibling(self):
        old=row("old",candidate=False,aliases=["derived"],uses=[dict(role="head_fit",ref="e")])
        alias=row("copy",canonical_id="derived")
        sibling=row("text",source_block=alias["source_block"],modality="text")
        counts,_=classify([old,alias,sibling],{"e"})
        self.assertEqual(counts["candidate_rows"]["ineligible"],2)

    def test_unverified_is_unknown(self):
        counts,_=classify([row("x",history_review=None),row("y",canonical_id=None),
                           row("z",uses=[dict(role="unlisted",ref="e")])],{"e"})
        self.assertEqual(counts["candidate_rows"]["unknown"],3)

    def test_documented_block_count(self):
        counts,_=classify([row("a"),row("b",source_block="block-a")],{"e"})
        self.assertEqual(counts["eligible_source_blocks"],1)
        self.assertEqual(counts["candidate_rows"]["eligible"],2)

    def test_historical_scope(self):
        manifest=Path(__file__).resolve().parents[1]/"certification/historical_manifest.json"
        counts,_=classify(*load_manifest(manifest))
        self.assertEqual(counts["candidate_rows"],dict(eligible=0,ineligible=10000,unknown=0))
        self.assertEqual(counts["reason_counts"]["head_fit"],2048)

if __name__=="__main__":
    unittest.main()
