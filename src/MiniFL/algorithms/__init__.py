from . import cocktailsgd, gd, marina
from .cocktailsgd import get_cocktailgd_master_and_clients
from .ef21 import get_ef21_master_and_clients
from .gd import get_gd_master_and_clients
from .interfaces import (
    Client,
    Master,
    run_algorithm_sequantially,
    run_algorithm_with_processes,
    run_algorithm_with_threads,
)
from .marina import get_marina_master_and_clients, get_permk_marina_master_and_clients
