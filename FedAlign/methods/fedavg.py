import torch
from .base import Base_Client, Base_Server

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,betas=(0.9, 0.999)  # Default betas used in Adam
)
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
