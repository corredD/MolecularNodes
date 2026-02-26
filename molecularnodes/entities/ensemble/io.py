from pathlib import Path
from .cellpack import CellPack
from .star import StarFile


def load_starfile(file_path, node_setup=True, world_scale=0.01):
    ensemble = StarFile.from_starfile(file_path)
    ensemble.create_object(
        name=Path(file_path).name, node_setup=node_setup, world_scale=world_scale
    )

    return ensemble


def load_cellpack(
    file_path,
    name="NewCellPackModel",
    node_setup=True,
    fraction: float = 0.1,
):
    # CellPack builds its Blender objects during initialization. Respect the
    # import options here so the instanced molecule objects get renderable GN styles.
    ensemble = CellPack(
        file_path,
        name=name,
        node_setup=node_setup,
        fraction=fraction,
    )
    return ensemble
