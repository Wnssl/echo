import torch


def get_shallow_mlp_head(dim_in, hidden_in=1, dim_out=1):
    print('使用回归器')
    regressor = torch.nn.Sequential(
        torch.nn.BatchNorm1d(dim_in),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(dim_in, dim_in),
        torch.nn.BatchNorm1d(dim_in),
        torch.nn.GELU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(dim_in, dim_out),
    )
    regressor[-1].bias.data[0] = 0.516

    return regressor
