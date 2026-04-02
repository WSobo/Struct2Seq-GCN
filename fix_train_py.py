with open("scripts/train.py", "r") as f:
    text = f.read()

# Replace train_epoch
new_train = """
        # We only want to compute loss on the regions we care about
        # chain_M comes natively from LigandMPNN (1 for residues to predict, 0 otherwise)
        if hasattr(batch['protein'], 'chain_M'):
            mask = batch['protein'].chain_M.bool()
        else:
            mask = torch.ones_like(batch['protein'].y).bool()
        
        if mask.sum() == 0:
            continue
            
        # Filter logits and targets by mask
        masked_logits = logits[mask]
        masked_targets = batch['protein'].y[mask]"""

text = text.replace("""        # We only want to compute loss on the regions we care about
        # chain_M comes natively from LigandMPNN (1 for residues to predict, 0 otherwise)
        mask = batch.chain_M.bool() if hasattr(batch, 'chain_M') else torch.ones_like(batch.y).bool()
        
        if mask.sum() == 0:
            continue
            
        # Filter logits and targets by mask
        masked_logits = logits[mask]
        masked_targets = batch.y[mask]""", new_train.strip())

# Replace evaluate
new_eval = """
            if hasattr(batch['protein'], 'chain_M'):
                mask = batch['protein'].chain_M.bool()
            else:
                mask = torch.ones_like(batch['protein'].y).bool()
                
            if mask.sum() == 0:
                continue
                
            masked_logits = logits[mask]
            masked_targets = batch['protein'].y[mask]"""

text = text.replace("""            mask = batch.chain_M.bool() if hasattr(batch, 'chain_M') else torch.ones_like(batch.y).bool()
            if mask.sum() == 0:
                continue
                
            masked_logits = logits[mask]
            masked_targets = batch.y[mask]""", new_eval.strip())

with open("scripts/train.py", "w") as f:
    f.write(text)
