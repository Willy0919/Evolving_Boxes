# --------------------------------------------------------
# Evolving boxes
# --------------------------------------------------------


__sets = {}

from datasets.detrac import detrac
from evb.config import cfg


# Set up detrac_<split>
for split in ['train', 'val', 'trainval', 'test']:
        name = 'detrac_{}'.format(split)
        __sets[name] = (lambda split=split: detrac(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""

    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))

    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
