# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr, LOGGER

from .val import RTDETRDataset, RTDETRValidator

from torch import nn
from torch import optim


class RTDETRTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an RT-DETR detection model.

    Notes:
        - F.grid_sample used in rt-detr does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Example:
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model='rtdetr-l.yaml', data='coco8.yaml', imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
        
    def build_dataset(self, img_path, mode='val', batch=None):
        """Build RTDETR Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def get_validator(self):
        """Returns a DetectionValidator for RTDETR model validation."""
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
    
    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate,
        momentum, weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        
        g = [], [], [], [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam
            
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                nl = fullname.split('.')[1]
                head = [str(22)]
                if nl in head:
                    if 'bias' in fullname or 'norm' in fullname:  # bias (no decay)
                            g[2].append(param)
                    elif isinstance(module, bn):  # weight (no decay)
                            g[1].append(param)
                    else:  # weight (with decay)
                            g[0].append(param)
        
        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
            # optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.95), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')
            
        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer
    