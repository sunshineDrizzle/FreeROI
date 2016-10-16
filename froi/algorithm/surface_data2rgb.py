import numpy as np


class Data2rgb(object):

    def __init__(self, overlays):
        """
        Get composite RGB according to overlays' data.

        :param overlays: List
            A instance of the List contains overlays which are needed to be rendered.
        :return:
        """

        if not isinstance(overlays, list):
            raise TypeError("The class Data2RGB's argument should be a list.")

        # Initialize some attributes
        self.vector_list = []  # The vector means that scalar_data are shown as a vector.
        self.alpha_list = []
        self.colormap_list = []

        # Get data from overlays
        for overlay in overlays:

            vector = overlay.get_data()
            vector = vector.clip(overlay.get_min(), overlay.get_max()).astype(np.float64)
            # normalize the scalar_data to [0, 255] linearly
            vector = (vector - vector.min())/(vector.max() - vector.min())*255

            self.vector_list.append(vector.astype(np.uint8))
            self.alpha_list.append(overlay.get_alpha() * 255)  # The scalar_data's alpha is belong to [0, 1].
            self.colormap_list.append(overlay.get_colormap())

        self.vertex_number = len(self.vector_list[0])

    def _red2yellow(self, index):
        """
        Return a RGBA array whose color ranges from red to yellow according to scalar_data and alpha.

        :param index:
            The index of self.vector_list, self.alpha_list and self.colormap_list
        :return:
        """

        rgba_array = np.zeros((self.vertex_number, 4), dtype=np.uint8)
        rgba_array[:, 0] = 255 * self.vector_list[index].clip(0, 1)
        rgba_array[:, 1] = self.vector_list[index]
        rgba_array[:, 3] = self.alpha_list[index] * self.vector_list[index].clip(0, 1)

        return rgba_array

    def _blue2cyanblue(self, index):
        """
        Return a RGBA array whose color ranges from blue to cyan blue according to scalar_data and alpha.

        :param index:
            The index of self.vector_list, self.alpha_list and self.colormap_list
        :return:
        """

        rgba_array = np.zeros((self.vertex_number, 4), dtype=np.uint8)
        rgba_array[:, 1] = self.vector_list[index]
        rgba_array[:, 2] = 255 * self.vector_list[index].clip(0, 1)
        rgba_array[:, 3] = self.alpha_list[index] * self.vector_list[index].clip(0, 1)

        return rgba_array

    def get_rgba_array(self, index):
        """
        Return a RGBA array according to scalar_data, alpha and colormap.

        :param index:
            The index of self.vector_list, self.alpha_list and self.colormap_list
        :return:
        """

        if self.colormap_list[index] == 'red2yellow':
            rgba_array = self._red2yellow(index)
        elif self.colormap_list[index] == 'blue2cyanblue':
            rgba_array = self._blue2cyanblue(index)
        else:
            raise RuntimeError("We have not implemented {} colormap at present!".format(self.colormap_list[index]))

        return rgba_array

    @staticmethod
    def alpha_composition(rgba_list):
        """Composite several rgba arrays into one."""

        if not len(rgba_list):
            raise ValueError('Input list cannot be empty.')
        if np.ndim(rgba_list[0]) != 2:
            raise ValueError('rgba_array must be 2D')

        zero_array = np.zeros((rgba_list[0].shape[0], 4))
        rgba_list.insert(0, zero_array)

        result = np.array(rgba_list[0][:, :3], dtype=np.float64)
        for i in range(1, len(rgba_list)):
            item = np.array(rgba_list[i], dtype=np.float64)
            alpha_channel = item[:, -1]
            alpha_channels = np.tile(alpha_channel, (3, 1)).T
            result = item[:, :3] * alpha_channels + result * (255 - alpha_channels)
            result /= 255
        result = result.astype(np.uint8)

        return result

    def get_rgb(self):
        """
        Get the final composite rgb array.

        :return:
        """

        rgba_list = []
        # complete rgba_list
        for index in range(len(self.vector_list)):
            rgba_array = self.get_rgba_array(index)
            rgba_list.append(rgba_array)

        return self.alpha_composition(rgba_list)

    def get_vertex_number(self):
        return self.vertex_number


if __name__ == "__main__":
    import sip
    import os

    sip.setapi("QString", 2)
    sip.setapi("QVariant", 2)

    from froi.core.dataobject import Hemisphere
    from mayavi import mlab

    surf_dir = r'/nfs/t1/nsppara/corticalsurface/fsaverage/surf'

    # model init
    surf1 = os.path.join(surf_dir, 'lh.white')
    s1 = os.path.join(surf_dir, 'lh.thickness')
    s2 = os.path.join(surf_dir, 'lh.curv')
    s3 = os.path.join(surf_dir, 'rh.thickness')

    h1 = Hemisphere(surf1)
    h1.load_overlay(s1)
    h1.load_overlay(s2)
    h1.load_overlay(s3)
    h1.overlay_list[0].set_colormap('red2yellow')
    h1.overlay_list[1].set_colormap('blue2cyanblue')
    h1.overlay_list[2].set_colormap('red2yellow')
    h1.overlay_list[0].set_alpha(1.0)
    h1.overlay_list[1].set_alpha(0.75)
    h1.overlay_list[2].set_alpha(0.25)

    # geo_data
    x, y, z, f, nn = h1.surf.x, h1.surf.y, h1.surf.z, h1.surf.faces, h1.surf.nn

    # rgb_data
    data2rgb = Data2rgb(h1.overlay_list)
    rgb_array = data2rgb.get_rgb()
    alpha_channel = np.ones((data2rgb.get_vertex_number(), 1), dtype=np.uint8)*255
    rgba_lut = np.c_[rgb_array, alpha_channel]
    scalars = np.array(range(data2rgb.get_vertex_number()))

    geo_mesh = mlab.pipeline.triangular_mesh_source(x, y, z, f, scalars=scalars)
    geo_mesh.data.point_data.normals = nn
    geo_mesh.data.cell_data.normals = None
    surf = mlab.pipeline.surface(geo_mesh)

    # surf.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgba_lut.shape[0])
    # surf.module_manager.scalar_lut_manager.lut.number_of_colors = rgba_lut.shape[0]
    surf.module_manager.scalar_lut_manager.lut.table = rgba_lut

    mlab.show()
    raw_input()
