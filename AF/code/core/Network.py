import os
import time
import torch
from torch import Tensor
from torchsummary import summary
from AF.code.core.settings import DEVICE
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self._device = DEVICE



    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network{}.txt".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time()))))), 'a+').write(str(self._network))
        open(os.path.join(path_to_log,
                          "network{}.txt".format(str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))))),
             'a+').write(str(summary(self, input_size=(2, 256, 256))))


    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        loss = nn.MSELoss()
        pred = pred.to(torch.float32)
        label = label.to(torch.float32)
        pred = pred.squeeze()
        label = label.squeeze()
        mse = loss(pred.to(self._device), label.to(self._device))
        rmse = mse**0.5
        return rmse



    def save(self, path_to_log: str, model_name):
        torch.save(self.state_dict(), os.path.join(path_to_log, model_name))

    def load(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained)
        self.load_state_dict(torch.load(path_to_model, map_location=self._device))
        #sd = torch.load(path_to_model, map_location=self._device)
        #print(sd.keys())
        #part_sd = {k: v for k, v in sd.items() if k not in ['conv1.0.weight', 'conv2.0.weight']}
        #self._network.load_state_dict(part_sd,strict=False)



