import json
from pathlib import Path
import re
import bpy
import numpy as np
from biotite.structure import AtomArray
from databpy import AttributeTypes, BlenderObject, store_named_attribute
from ... import blender as bl
from ... import color
from ...nodes import nodes
from ..utilities import create_object
from .base import Ensemble, EntityType
from .reader import CellPackReader


class CellPack(Ensemble):
    def __init__(
        self,
        file_path,
        name: str | None = None,
        node_setup: bool = True,
        fraction: float = 0.1,
        as_points: bool = False,
    ):
        super().__init__(file_path)
        self._entity_type = EntityType.ENSEMBLE_CELLPACK
        self.file_type = self._file_type()
        self.file = CellPackReader(file_path)
        self.transformations = self.file.get_assemblies()
        self.color_entity = {}
        self._color_palette_path = Path(file_path).parent / "color_palette.json"
        object_name = name if name is not None else f"{Path(file_path).name}"
        self._create_object_instances(name=object_name, node_setup=node_setup)
        self.object = self._create_data_object(name=object_name)
        self.object.mn.entity_type = self._entity_type.value
        self._debug_instance_collection_order()
        self._setup_node_tree(fraction=fraction, as_points=as_points)
        # self._setup_colors()

    def _setup_colors(self):
        if self._color_palette_path.exists():
            self.color_palette = json.load(open(self._color_palette_path))

        for entity in np.unique(self.array.entity_id):
            ename = self.data.entities[entity]
            if ename in self.color_palette:
                rgb = [self.color_palette[ename][c] / 255.0 for c in "xyz"]
                self.color_entity[entity] = np.array([*rgb, 1.0])
            else:
                self.color_entity[entity] = color.random_rgb(int(entity))

            self.entity_chains[entity] = (
                np.unique(self.array.asym_id[self.array.entity_id == entity]) @ property
            )

    @property
    def molecules(self):
        return self.file.molecules

    def _file_type(self):
        return Path(self.file_path).suffix.strip(".")

    @staticmethod
    def _debug_print_rows(title: str, rows: list[str], limit: int = 25) -> None:
        print(title)
        for row in rows[:limit]:
            print(f"  {row}")
        if len(rows) > limit:
            print(f"  ... {len(rows) - limit} more")

    @classmethod
    def _debug_print_items(cls, title: str, items, limit: int = 25) -> None:
        rows = [f"[{idx}] {item}" for idx, item in enumerate(list(items))]
        cls._debug_print_rows(title, rows, limit=limit)

    @staticmethod
    def _hide_source_chain_object(obj: bpy.types.Object) -> None:
        # Keep source chains available for instancing, but out of view.
        try:
            obj.hide_set(True)
        except RuntimeError:
            pass
        obj.hide_render = True

    @staticmethod
    def _blender_name_sort_key(name: str):
        # Match Collection Info + Pick Instance ordering for child objects.
        parts = re.split(r"(\d+)", str(name))
        key = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append((1, int(part)))
            else:
                key.append((0, part.casefold()))
        return tuple(key)

    def _instance_mol_ids_for_collection(self) -> np.ndarray:
        if hasattr(self, "_instance_mol_ids_cache"):
            return self._instance_mol_ids_cache

        all_mol_ids = np.asarray(self.file.mol_ids).astype(str)
        transform_chain_ids = np.asarray(self.transformations["chain_id"]).astype(str)
        transform_chain_id_set = set(transform_chain_ids.tolist())
        molecule_lookup = {str(mol_id) for mol_id in self.molecules.keys()}

        # Keep the collection order identical to the object creation order.
        instance_mol_ids = np.asarray(
            [mol_id for mol_id in all_mol_ids if mol_id in transform_chain_id_set],
            dtype=str,
        )

        if len(instance_mol_ids) == 0:
            # Some formats (e.g. PETWORLD variants) can use a different naming scheme
            # between assembly transform chain IDs and molecule object keys.
            print(
                "CellPack: no overlap between transform chain_ids and molecule keys; "
                "falling back to all molecule IDs"
            )
            instance_mol_ids = all_mol_ids

        skipped_mol_ids = [mol_id for mol_id in all_mol_ids if mol_id not in set(instance_mol_ids)]
        if skipped_mol_ids:
            print(
                "CellPack: skipping molecules without transforms:",
                skipped_mol_ids,
            )

        missing_molecule_ids = [
            chain_id
            for chain_id in sorted(transform_chain_id_set)
            if chain_id not in molecule_lookup
        ]
        if missing_molecule_ids:
            print(
                "CellPack: transform chain_ids without matching molecule objects:",
                missing_molecule_ids,
            )

        self._instance_mol_ids_cache = instance_mol_ids
        return self._instance_mol_ids_cache

    def _instance_ids_for_mapping(self) -> np.ndarray:
        if hasattr(self, "_instance_collection_name"):
            collection_ids = np.asarray(
                [obj.name for obj in self.instance_collection.objects]
            ).astype(str)
        else:
            collection_ids = self._instance_mol_ids_for_collection()

        return np.asarray(
            sorted(collection_ids.tolist(), key=self._blender_name_sort_key),
            dtype=str,
        )

    def _transform_instance_indices(self) -> tuple[np.ndarray, np.ndarray]:
        instance_mol_ids = self._instance_ids_for_mapping()

        # Convert transform keys into collection slot indices before GN sees them.
        if self.file._is_petworld:
            transform_model_nums = np.asarray(self.transformations["pdb_model_num"]).astype(int)
            model_num_lookup = {}
            rows = []
            for idx, mol_id in enumerate(instance_mol_ids):
                model_num = int(np.asarray(self.molecules[mol_id].pdb_model_num).astype(int)[0])
                if model_num in model_num_lookup:
                    raise ValueError(
                        "CellPack PETWORLD molecules share the same model number: "
                        f"{model_num}"
                    )
                model_num_lookup[model_num] = idx
                rows.append(f"model {model_num} -> slot[{idx}] {mol_id}")

            self._debug_print_rows(
                "CellPack PETWORLD model_num -> instance slot",
                rows,
            )

            missing_model_nums = sorted(
                {int(model_num) for model_num in transform_model_nums if model_num not in model_num_lookup}
            )
            if missing_model_nums:
                raise ValueError(
                    "CellPack PETWORLD transform rows reference missing model numbers: "
                    f"{missing_model_nums}"
                )

            chain_id_int = np.array(
                [model_num_lookup[int(model_num)] for model_num in transform_model_nums],
                dtype=int,
            )
            return chain_id_int, instance_mol_ids

        transform_chain_ids = np.asarray(self.transformations["chain_id"]).astype(str)
        chain_id_lookup = {
            chain_name: idx for idx, chain_name in enumerate(instance_mol_ids.tolist())
        }
        self._debug_print_rows(
            "CellPack chain_id -> instance slot",
            [f"{chain_name} -> slot[{idx}]" for chain_name, idx in chain_id_lookup.items()],
        )

        missing_chain_ids = sorted(
            {chain_name for chain_name in transform_chain_ids if chain_name not in chain_id_lookup}
        )
        if missing_chain_ids:
            raise ValueError(
                "CellPack transform rows reference missing molecule objects: "
                f"{missing_chain_ids}"
            )

        chain_id_int = np.array(
            [chain_id_lookup[chain_name] for chain_name in transform_chain_ids],
            dtype=int,
        )
        return chain_id_int, instance_mol_ids

    def _assign_colors(self, obj: bpy.types.Object, array: AtomArray):
        # random color per chain
        # could also do by entity, + chain-lighten + atom-lighten

        entity = array.entity_id[0]
        color_entity = self.color_entity[entity]
        nc = len(self.entity_chains[entity])
        ci = np.where(self.entity_chains[entity] == array.chain_name)[0][0] * 2
        color_chain = color.Lab.lighten_color(color_entity, (float(ci) / nc))
        colors = np.tile(color_chain, (len(array), 1))

        store_named_attribute(
            obj=obj,
            name="Color",
            data=colors,
            atype=AttributeTypes.FLOAT_COLOR,
        )

    def _create_object_instances(
        self, name: str = "CellPack", node_setup: bool = True
    ) -> bpy.types.Collection:
        collection = bl.coll.cellpack(name)
        instance_mol_ids = self._instance_mol_ids_for_collection()

        for i, mol_id in enumerate(instance_mol_ids):
            array = self.molecules[mol_id]
            print(f"CellPack instance slot[{i}]: {mol_id} ({len(array)} atoms)")
            obj = create_object(
                array=array,
                name=mol_id,
                collection=collection,
            )
            obj.mn.entity_type = self._entity_type.value
            self._hide_source_chain_object(obj)

            if len(self.color_entity) > 0:
                self._assign_colors(obj, array)

            if node_setup:
                nodes.create_starting_node_tree(
                    obj,
                    name=f"MN_pack_instance_{name}",
                    color=None,
                    material="MN Ambient Occlusion",
                )

        self.data_collection = collection
        self.instance_collection = collection
        print("CellPack: source chain objects hidden; only the instanced ensemble stays visible")

        return collection

    def _debug_instance_collection_order(self) -> None:
        actual_ids = np.asarray([obj.name for obj in self.instance_collection.objects]).astype(str)
        collection_info_ids = self._instance_ids_for_mapping()

        self._debug_print_items("CellPack collection.objects order", actual_ids)
        self._debug_print_items("CellPack Collection Info instance order", collection_info_ids)

        if np.array_equal(actual_ids, collection_info_ids):
            print("CellPack: collection.objects order matches Collection Info ordering")
        else:
            print("CellPack: Collection Info reorders child instances for Pick Instance")

    def _create_data_object(self, name="DataObject"):
        chain_id_int, instance_mol_ids = self._transform_instance_indices()

        bob = BlenderObject(
            bl.mesh.create_data_object(
                self.transformations, name=name, collection=bl.coll.mn()
            )
        )
        bob.object["chain_ids"] = instance_mol_ids.tolist()
        bob.store_named_attribute(
            data=chain_id_int,
            name="chain_id",
            atype=AttributeTypes.INT,
        )
        self._debug_print_items("CellPack stored chain_ids", instance_mol_ids)

        return bob.object

    def _create_transparent_material(self, name="MN Transparent"):
        # Create a new material
        material_name = name
        material = bpy.data.materials.new(name=material_name)

        # Enable 'Use Nodes'
        material.use_nodes = True

        # Clear all default nodes
        nodes = material.node_tree.nodes
        nodes.clear()

        # Add a Material Output node
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (300, 0)

        # Add a Transparent BSDF node
        transparent_node = nodes.new(type="ShaderNodeBsdfTransparent")
        transparent_node.location = (0, 0)

        # Connect the Transparent BSDF node to the Material Output node
        material.node_tree.links.new(
            transparent_node.outputs["BSDF"], output_node.inputs["Surface"]
        )

        # Optionally set the color of the transparent BSDF
        transparent_node.inputs["Color"].default_value = (1, 1, 1, 1)  # RGBA
        return material

    def _setup_node_tree(self, name="CellPack", fraction=1.0, as_points=False):
        mod = nodes.get_mod(self.object)

        group = nodes.new_tree(name=f"MN_ensemble_{name}", fallback=False)
        mod.node_group = group

        node_pack = nodes.add_custom(group, "Ensemble Instance", location=[-100, 0])
        node_pack.inputs["Instances"].default_value = self.data_collection
        # node_pack.inputs["Fraction"].default_value = fraction
        # node_pack.inputs["As Points"].default_value = as_points

        # Create the GeometryNodeIsViewport node
        node_is_viewport = group.nodes.new("GeometryNodeIsViewport")
        node_is_viewport.location = (-490.0, -240.0)

        # Create the GeometryNodeSwitch node
        node_switch = group.nodes.new("GeometryNodeSwitch")
        node_switch.location = (-303.0, -102.0)
        # Set the input type of the switch node to FLOAT
        node_switch.input_type = "FLOAT"
        # Set the true and false values of the switch node
        node_switch.inputs[1].default_value = 1.0
        node_switch.inputs[2].default_value = 0.1

        group.links.new(node_is_viewport.outputs[0], node_switch.inputs[0])

        group.links.new(node_switch.outputs[0], node_pack.inputs["Fraction"])

        group.links.new(node_is_viewport.outputs[0], node_pack.inputs["As Points"])

        # createa a plane primitive node
        node_plane = group.nodes.new("GeometryNodeMeshGrid")
        node_plane.location = (-1173, 252)

        # create a geomtry transform node
        node_transform = group.nodes.new("GeometryNodeTransform")
        node_transform.location = (-947, 245)
        # change mesh translation
        # node_transform.inputs[1].default_value = (3.0, 0.0, 0.0)
        # # change mesh rotation
        # node_transform.inputs[2].default_value = (0.0, 3.14 / 2.0, 0.0)
        # # change mesh scale
        # node_transform.inputs[3].default_value = (50.0, 50.0, 1.0)
        # link the plane to the transform node
        group.links.new(node_plane.outputs[0], node_transform.inputs[0])

        # create transparent material and setMaterial node
        material = self._create_transparent_material()
        node_set_material = group.nodes.new("GeometryNodeSetMaterial")
        node_set_material.location = (-100, 289)
        group.links.new(node_transform.outputs[0], node_set_material.inputs[0])
        node_set_material.inputs[2].default_value = material
        # create the join geoemtry node
        node_join = group.nodes.new("GeometryNodeJoinGeometry")
        node_join.location = (151, 122)
        group.links.new(node_set_material.outputs[0], node_join.inputs[0])
        group.links.new(node_pack.outputs[0], node_join.inputs[0])

        # create a geomtry proximity node and link the plane to it
        node_proximity = group.nodes.new("GeometryNodeProximity")
        node_proximity.location = (-586, 269)
        group.links.new(node_transform.outputs[0], node_proximity.inputs[0])

        # get the position attribute node
        node_position = group.nodes.new("GeometryNodeInputPosition")
        node_position.location = (-796, 86)

        # link it to the posistion sample in proximity
        group.links.new(node_position.outputs[0], node_proximity.inputs[2])

        # create a compare node that take the distance from the proximity node
        # and compare it to be greter than 2.0
        node_compare = group.nodes.new("FunctionNodeCompare")
        node_compare.location = (-354, 316)
        node_compare.data_type = "FLOAT"
        node_compare.operation = "GREATER_THAN"
        node_compare.inputs[1].default_value = 2.0
        # do the link
        group.links.new(node_proximity.outputs[1], node_compare.inputs[0])

        # link the outpot of the compare node to the selection node_pack
        group.links.new(node_compare.outputs[0], node_pack.inputs["Selection"])
        
        link = group.links.new
        link(nodes.get_input(group).outputs[0], node_pack.inputs[0])
        # link(node_pack.outputs[0], nodes.get_output(group).inputs[0])
        link(node_join.outputs[0], nodes.get_output(group).inputs[0])
