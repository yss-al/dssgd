import torch
import torch.nn as nn


class MLPMnist(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim_in, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, dim_out),
                                    nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x.view(-1, x.nelement()//x.shape[0])
        x = self.layers(x)
        return x
    
    def download(self, w_list):
        w_dict = {}
        start = 0
        for k, v in self.state_dict().items():
            w_dict[k] = w_list[start:start + v.nelement()].view(v.size())
            print(f'{k}: {v.nelement()}')
            print(w_dict[k].size())
            start += v.nelement()
            print(start)
        self.load_state_dict(w_dict)


    def train_one_epoch(self):
        print('one epoch')
        pass

    def upload_paras(self, theta=0.1):
        grad_list = [p.grad for p in self.parameters()]
        grad_glob = torch.cat(grad_list)
        _, indexes = torch.topk(len(grad_glob) * theta)
        grad_zero = torch.zeros_like(grad_glob)
        grad_zero[indexes] = 1.
        return grad_zero * grad_glob

    def parameters_to_list(self):
        grad_list = [v.flatten() for _, v in self.state_dict().items()]
        return torch.cat(grad_list)
