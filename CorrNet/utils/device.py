import os
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)
        if device.lower() == 'cpu':
            print("Using CPU")
            self.gpu_list = []
            self.output_device = torch.device('cpu')
        else:
            self.gpu_list = [i for i in range(len(device.split(',')))]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            self.output_device = torch.device(f'cuda:{self.gpu_list[0]}')
            self.occupy_gpu(self.gpu_list)

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if gpus is None or len(gpus) == 0:
            print("No GPUs specified or using CPU")
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                if torch.cuda.is_available():
                    torch.zeros(1).cuda(g)

