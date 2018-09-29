import torch


def get_model(model, checkpoint=None, map_location=None, devices=None):
    model.cuda()

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location)#.module.state_dict()
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd}
        print('Overlapped keys: {}'.format(len(sd.keys())))
        msd.update(sd)
        model.load_state_dict(msd)

    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices)

    return model


def freeze(model, unfreeze=False):
    children = list(model.children())
    if hasattr(model, 'children') and len(children):
        for child in children:
            freeze(child, unfreeze)
    elif hasattr(model, 'parameters'):
        for param in model.parameters():
            param.requires_grad = unfreeze


def unfreeze_bn(model):
    if isinstance(model, torch.nn.BatchNorm2d):
        for param in model.parameters():
            param.requires_grad = True

    children = list(model.children())
    if len(children):
        for child in children:
            unfreeze_bn(child)
    return None
