import numpy as np


def _red2yellow(vector, alpha):
    """
    Return a RGBA array whose color ranges from red to yellow according to scalar_data and alpha.

    :param vector:
    :param alpha:
    :return:
    """

    vertex_number = len(vector)

    rgba_array = np.zeros((vertex_number, 4), dtype=np.uint8)
    rgba_array[:, 0] = 255 * vector.clip(0, 1)
    rgba_array[:, 1] = vector
    rgba_array[:, 3] = alpha * vector.clip(0, 1)

    return rgba_array


def _blue2cyanblue(vector, alpha):
    """
    Return a RGBA array whose color ranges from blue to cyan blue according to scalar_data and alpha.

    :param vector:
    :param alpha:
    :return:
    """

    vertex_number = len(vector)

    rgba_array = np.zeros((vertex_number, 4), dtype=np.uint8)
    rgba_array[:, 1] = vector
    rgba_array[:, 2] = 255 * vector.clip(0, 1)
    rgba_array[:, 3] = alpha * vector.clip(0, 1)

    return rgba_array


def _normalize255(vector, vmin, vmax):
    """
    normalize the scalar_data to [0, 255] linearly

    :param vector:
    :param vmin:
    :param vmax:
    :return:
    """

    vector = vector.clip(vmin, vmax).astype(np.float64)
    vector = (vector - vector.min())/(vector.max() - vector.min())*255

    return vector.astype(np.uint8)


def get_rgba_array(overlay):
    """
    Return a RGBA array according to scalar_data, alpha and colormap.
    """

    colormap = overlay.get_colormap()
    vector = overlay.get_data()
    vector = _normalize255(vector, overlay.get_min(), overlay.get_max())
    alpha = overlay.get_alpha() * 255  # The scalar_data's alpha is belong to [0, 1].

    if colormap == 'red2yellow':
        rgba_array = _red2yellow(vector, alpha)
    elif colormap == 'blue2cyanblue':
        rgba_array = _blue2cyanblue(vector, alpha)
    else:
        raise RuntimeError("We have not implemented {} colormap at present!".format(self.colormap_list[index]))

    return rgba_array


def alpha_composition(rgba_list):
    """Composite several rgba arrays into one."""

    if not len(rgba_list):
        raise ValueError('Input list cannot be empty.')
    if np.ndim(rgba_list[0]) != 2:
        raise ValueError('rgba_array must be 2D')

    zero_array = np.zeros(rgba_list[0].shape)
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


if __name__ == "__main__":
    import sip
    import os

    sip.setapi("QString", 2)
    sip.setapi("QVariant", 2)

    from froi.core.dataobject import Hemisphere
    from mayavi import mlab

    surf_dir = r'/nfs/t1/nsppara/corticalsurface/fsaverage/surf'

    # model init
    surf1 = os.path.join(surf_dir, 'rh.white')
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
    rgb_array = h1.get_composited_rgb()
    vertex_number = rgb_array.shape[0]
    alpha_channel = np.ones((vertex_number, 1), dtype=np.uint8)*255
    rgba_lut = np.c_[rgb_array, alpha_channel]
    scalars = np.array(range(vertex_number))

    geo_mesh = mlab.pipeline.triangular_mesh_source(x, y, z, f, scalars=scalars)
    geo_mesh.data.point_data.normals = nn
    geo_mesh.data.cell_data.normals = None
    surf = mlab.pipeline.surface(geo_mesh)

    # surf.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgba_lut.shape[0])
    # surf.module_manager.scalar_lut_manager.lut.number_of_colors = rgba_lut.shape[0]
    surf.module_manager.scalar_lut_manager.lut.table = rgba_lut

    mlab.show()
    raw_input()
