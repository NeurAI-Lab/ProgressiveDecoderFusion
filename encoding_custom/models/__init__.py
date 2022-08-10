from .uninet_hierarchical import get_uninet_hierarchical


def get_multitask_model(name, **kwargs):
    models = {'uninet_hierarchical': get_uninet_hierarchical}
    return models[name.lower()](**kwargs)
