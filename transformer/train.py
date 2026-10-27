import torch
from torch import Tensor
from transformer_zsq import Transformer, shift_targets, make_padding_mask

def train(
        model,
        optimizer,
        src_token_ids,
        tgt_token_ids,
        tgt_pad_id
):
    model.train()
    optimizer.zero_grad()
    decoder_input, labels = shift_targets(tgt_token_ids)
    logits, _, _, _ = model(src_token_ids, decoder_input)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), 
        labels.reshape(-1), 
        ignore_index=tgt_pad_id)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def decode_greedy(
    model,
    bos_id,
    eos_id,
    max_new_tokens,
    src_token_ids, # [B, L]
):
    model.eval()
    src_mask = make_padding_mask(src_token_ids, model.src_pad_id)
    memory,_ = model.encoder(src_token_ids, src_mask)

    B = src_token_ids.shape[0]
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=src_token_ids.device)
    finished = torch.zeros(B, dtype=torch.bool, device=src_token_ids.device)

    for _ in range(max_new_tokens):
        decoder_output, _, _ = model.decoder(generated, memory, src_mask) # [B, T, D]
        last_hidden_state = decoder_output[:, -1, :] # [B, D]
        last_logits = model.out_proj(last_hidden_state) # [B, vocab]
        last_token = last_logits.argmax(dim=-1) # [B]
        last_token = torch.where(finished, torch.full_like(last_token, eos_id), last_token) # [B]
        generated = torch.cat([generated, last_token.unsqueeze(-1)], dim=-1)
        finished = finished | last_token.eq(eos_id)
        if finished.all().item():
            break
    return generated

    
def test_overfit_and_generate():
    device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)

    print("device:", device)
    if device.type == "cuda":
        print("device name:", torch.cuda.get_device_name(device))

    torch.manual_seed(42)

    pad_id = 0
    bos_id = 1
    eos_id = 2

    src_token_ids = torch.tensor([[1, 2, 3, pad_id]], device=device)
    tgt_token_ids = torch.tensor([[bos_id, 4, 5, 6, eos_id]], device=device)
    model = Transformer(
        src_pad_id = pad_id,
        tgt_pad_id = pad_id,
        src_vocab_size = 16,
        tgt_vocab_size = 16,
        max_len = 16,
        d_model = 32,
        d_ff = 64,
        num_head = 4,
        num_layers = 2,
        dropout = 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for step in range(1000):
        loss = train(
            model = model,
            optimizer = optimizer,
            src_token_ids = src_token_ids,
            tgt_token_ids = tgt_token_ids,
            tgt_pad_id = pad_id)
        if step % 100 == 0:
            print(f'Step:{step}, loss={loss}')

    generated = decode_greedy(
        model=model,
        bos_id=bos_id,
        eos_id=eos_id,
        max_new_tokens=16,
        src_token_ids=src_token_ids
    )
    print(f'generated:{generated}')

if __name__ == "__main__":
    test_overfit_and_generate()