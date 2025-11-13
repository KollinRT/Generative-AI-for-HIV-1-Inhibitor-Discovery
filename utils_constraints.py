from transformers import LogitsProcessor
import torch
from typing import List


class _Tok:
    def __init__(self, tokenizer):
        self.tok = tokenizer

        # resolve ids once
        def get_id(t):
            i = tokenizer.convert_tokens_to_ids(t)
            return None if i is None or i == tokenizer.unk_token_id else i

        self.BRANCH = {t: get_id(t) for t in ("[Branch1]", "[Branch2]")}
        self.RING = {
            t: get_id(t) for t in ("[Ring1]", "[Ring2]", "[=Ring1]", "[=Ring2]")
        }
        self.ATTACH = set(x for x in self.BRANCH.values() if x is not None)
        self.RINGS = set(x for x in self.RING.values() if x is not None)
        # atoms
        self.Npos = get_id("[N+1]")
        self.N = get_id("[N]")
        self.Sdbl = get_id("[=S]")
        self.Ndbl = get_id("[=N]")
        self.halogens = {h: get_id(h) for h in ("[F]", "[Cl]", "[Br]", "[I]")}
        self.EOS = tokenizer.eos_token_id


def _endswith(seq: List[int], suffix: List[int]) -> bool:
    if len(seq) < len(suffix):
        return False
    if not suffix:
        return True
    return seq[-len(suffix) :] == suffix


class MonovalentHalogenProcessor(LogitsProcessor):
    """
    Disallow adding another branch/attachment immediately after placing a halogen.
    Approximate: if last token is halogen atom token, mask branch tokens.
    """

    def __init__(self, tokenizer):
        self.T = _Tok(tokenizer)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last = input_ids[:, -1]
        for b in range(input_ids.size(0)):
            if int(last[b]) in self.T.halogens.values():
                scores[b, list(self.T.ATTACH)] = -1e9
        return scores


class NitreniumBranchCapProcessor(LogitsProcessor):
    """
    Pattern mask: if we have ... [N+1] [=S]  (or [=N]) and already opened 2 attachments,
    forbid a 3rd branch/attachment. We count only immediate branch tokens after that pattern.
    """

    def __init__(self, tokenizer):
        self.T = _Tok(tokenizer)
        self.patterns = []
        if self.T.Npos is not None and self.T.Sdbl is not None:
            self.patterns.append([self.T.Npos, self.T.Sdbl])
        if self.T.Npos is not None and self.T.Ndbl is not None:
            self.patterns.append([self.T.Npos, self.T.Ndbl])

    def __call__(self, input_ids, scores):
        for b in range(input_ids.size(0)):
            seq = input_ids[b].tolist()
            # find last occurrence of pattern
            k = -1
            for p in self.patterns:
                for i in range(len(seq) - len(p)):
                    if seq[i : i + len(p)] == p:
                        k = max(k, i + len(p))
            if k >= 0:
                # count branch tokens after k
                post = seq[k:]
                used = sum(1 for t in post if t in self.T.ATTACH)
                if used >= 2:
                    scores[b, list(self.T.ATTACH)] = -1e9
        return scores


class RingBalanceProcessor(LogitsProcessor):
    """
    Prevent opening absurd number of rings and force close balance lightly:
    - cap total simultaneously open rings (per index) to 1
    - if too many opens, mask further opens until a close appears
    """

    def __init__(self, tokenizer, max_open=2):
        self.T = _Tok(tokenizer)
        self.max_open = max_open

    def __call__(self, input_ids, scores):
        for b in range(input_ids.size(0)):
            seq = input_ids[b].tolist()
            opens = 0
            for t in seq:
                if t in (self.T.RING.get("[Ring1]"), self.T.RING.get("[Ring2]")):
                    opens += 1
                elif t in (self.T.RING.get("[=Ring1]"), self.T.RING.get("[=Ring2]")):
                    opens = max(0, opens - 1)
            if opens >= self.max_open:
                # mask more ring opens
                ids_to_mask = [self.T.RING.get("[Ring1]"), self.T.RING.get("[Ring2]")]
                ids_to_mask = [i for i in ids_to_mask if i is not None]
                scores[b, ids_to_mask] = -1e9
        return scores
