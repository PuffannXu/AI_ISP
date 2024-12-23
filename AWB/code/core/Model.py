import os
import time
import torch
from torch import Tensor
from torchsummary import summary
from AWB.code.core.settings import DEVICE



class Model:
    def __init__(self):
        self._device = DEVICE

        self._optimizer = None
        self._network = None

    def print_network(self):
        print("\n----------------------------------------------------------\n")
        print(self._network)
        print("\n----------------------------------------------------------\n")


    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network{}.txt".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time()))))), 'a+').write(str(self._network))
        open(os.path.join(path_to_log,
                          "network{}.txt".format(str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))))),
             'a+').write(str(summary(self._network, input_size=(3, 256, 256))))

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_log: str):
        torch.save(self._network.state_dict(), os.path.join(path_to_log, "model.pth"))

    def load(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained)
        self._network.load_state_dict(torch.load(path_to_model, map_location=self._device))
        #sd = torch.load(path_to_model, map_location=self._device)
        #print(sd.keys())
        #part_sd = {k: v for k, v in sd.items() if k not in ['conv1.0.weight', 'conv2.0.weight']}
        #self._network.load_state_dict(part_sd,strict=False)


    def set_optimizer(self, learning_rate: float, optimizer_type: str = "adam"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop,"sgd":torch.optim.SGD}
        if optimizer_type=="adam":
            self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate, betas=(0.8, 0.999), eps=1e-08, weight_decay=0)
        elif optimizer_type=="rmsprop":
            self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)
        else:
            self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate, momentum=0.8, nesterov=True)


