from .mount_model import MountModel
from .mount_factory import mount_factory

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount
from .fetch_mount import FetchMount


MOUNT_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "FetchMount": FetchMount,
    None: NullMount,
}

ALL_MOUNTS = MOUNT_MAPPING.keys()
