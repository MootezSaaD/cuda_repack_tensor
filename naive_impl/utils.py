import torch
from torch.nn.utils.rnn import pad_sequence

def repack_tensor_and_create_mask(tensor, mask, fuse = False):
    """
    Given a `mask`, this function removes from `tensor` the tokens according to that mask and returns
    the new batch tensor and updated mask.
    If `fuse` is True, it will merge the masked tokens into one tensor which will be included in the new sequence.
    """
    batch = []
    lengths = []
    for el, msk in zip(tensor, mask):
        new_len = msk.sum().item()
        if fuse:
          new_len += 1
        _, hidden_dim = el.shape
        _m = msk[..., None].bool()
        if fuse:
          new_el = el.masked_select(_m)
          inv_m = ~_m
          num_masked_tokens = inv_m.int().sum().item()
          fused_tokens = el.masked_select(inv_m).reshape((num_masked_tokens, hidden_dim)).mean(0)
          new_el = torch.cat((new_el, fused_tokens)).reshape((new_len, hidden_dim))
        else:
          new_el = el.masked_select(_m).reshape((new_len, hidden_dim))
        batch.append(new_el)
        lengths.append(new_len)
    
    padded_batch = pad_sequence(batch, batch_first=True)
    new_mask = torch.zeros_like(padded_batch[..., 0], dtype=torch.bool)
    for i, L in enumerate(lengths):
        new_mask[i, :L] = True
    
    return padded_batch, new_mask