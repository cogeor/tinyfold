import torch
ckpt = torch.load('outputs/overfit_rotation_long/best_model.pt', map_location='cpu')
print("Keys:", ckpt.keys())
for k in ckpt.keys():
    if k != 'model_state_dict':
        print(f"  {k}: {ckpt[k]}")
