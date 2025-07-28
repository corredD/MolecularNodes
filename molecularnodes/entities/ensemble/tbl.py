from pathlib import Path

import bpy
import mrcfile
import numpy as np
import dynamotable
from PIL import Image

from ... import blender as bl
from .ensemble import Ensemble


class TBLFile(Ensemble):
    def __init__(self, file_path, pixel_size=1.0):
        super().__init__(file_path)
        self.type = "tblfile"
        self.pixel_size = pixel_size

    @classmethod
    def from_tblfile(cls, file_path, pixel_size=1.0):
        self = cls(file_path, pixel_size=pixel_size)
        self.data = self._read()
        self.positions = None
        self.current_image = -1
        self._create_mn_columns()
        self.n_images = self._n_images()
        return self

    @classmethod
    def from_blender_object(cls, blender_object, pixel_size=1.0):
        self = cls(blender_object["tblfile_path"], pixel_size=pixel_size)
        self.object = blender_object
        self.data = self._read()
        self.positions = None
        self.current_image = -1
        self._create_mn_columns()
        self.n_images = self._n_images()
        bpy.app.handlers.depsgraph_update_post.append(self._update_micrograph_texture)
        return self

    @property
    def star_node(self):
        return bl.nodes.get_star_node(self.object)

    @property
    def micrograph_material(self):
        return bl.nodes.MN_micrograph_material()

    def _read(self):
        star = dynamotable.read(self.file_path)
        return star

    def _n_images(self):
        if isinstance(self.data, dict):
            return len(self.data)
        return 1

    def _create_mn_columns(self):
        df = self.data
        # get necessary info from dataframes
        # 24  : x             x coordinate in original volume
        # 25  : y             y coordinate in original volume
        # 26  : z             z coordinate in original volume
        self.positions = np.column_stack([df.x.values, df.y.values, df.z.values])
        pixel_size = self.pixel_size
        # 4   : dx            x shift from center (in pixels)
        # 5   : dy            y shift from center (in pixels)
        # 6   : dz            z shift from center (in pixels)
        dx = np.column_stack([df.dx.values, df.dy.values, df.dz.values])
        self.positions += dx / pixel_size
        self.positions = self.positions * pixel_size
        # 7   : tdrot         euler angle (rotation clockwise around z, in degrees), (for tilt direction gets rotated)
        # 8   : tilt          euler angle (rotation clockwise around new x, in degrees)
        # 9   : narot         euler angle (rotation clockwise around new z, in degrees), (for new azymuthal rotation)
        df["MNAnglePhi"] = df.tdrot
        df["MNAngleTheta"] = df.tilt
        df["MNAnglePsi"] = df.narot
        df["MNPixelSize"] = pixel_size
        df["MNImageId"] = 0.0
        self.data = df

    def create_object(self, name="TBLFileObject", node_setup=True, world_scale=0.01):
        blender_object = bl.mesh.create_object(
            self.positions * world_scale, collection=bl.coll.mn(), name=name
        )

        blender_object.mn["molecule_type"] = "tbl"

        # create attribute for every column in the STAR file
        for col in self.data.columns:
            col_type = self.data[col].dtype
            # If col_type is numeric directly add
            if np.issubdtype(col_type, np.number):
                bl.mesh.store_named_attribute(
                    blender_object,
                    col,
                    self.data[col].to_numpy().reshape(-1),
                    "FLOAT",
                    "POINT",
                )

            # If col_type is object, convert to category and add integer values
            elif col_type == object:
                codes = (
                    self.data[col].astype("category").cat.codes.to_numpy().reshape(-1)
                )
                bl.mesh.store_named_attribute(
                    blender_object, col, codes, "INT", "POINT"
                )
                # Add the category names as a property to the blender object
                blender_object[f"{col}_categories"] = list(
                    self.data[col].astype("category").cat.categories
                )
        blender_object.mn.uuid = self.uuid

        if node_setup:
            bl.nodes.create_starting_nodes_tblfile(
                blender_object, n_images=self.n_images
            )

        blender_object["tblfile_path"] = str(self.file_path)
        self.object = blender_object
        return blender_object
