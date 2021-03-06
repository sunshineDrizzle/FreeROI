import sys

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from tvtk.api import tvtk
from PyQt4.QtGui import *
from PyQt4 import QtCore
from mayavi.core.ui.api import SceneEditor, MayaviScene, MlabSceneModel
from mayavi import mlab
import numpy as np

from froi.widgets.treemodel import TreeModel
from froi.algorithm.tools import toggle_color, bfs
from froi.algorithm.meshtool import get_n_ring_neighbor, get_vtx_neighbor
from froi.algorithm.array2qimage import array2qrgba, _normalize255
from froi.core.labelconfig import LabelConfig


# Helpers
# ---------------------------------------------------------------------------------------------------------
def _toggle_toolbar(figure, show=None):
    """
    Toggle toolbar display

    Parameters
    ----------
    figure: the mlab figure
    show : bool | None
        If None, the state is toggled. If True, the toolbar will
        be shown, if False, hidden.
    """

    if figure.scene is not None:
        if hasattr(figure.scene, 'scene_editor'):
            # Within TraitsUI
            bar = figure.scene.scene_editor._tool_bar
        else:
            # Mayavi figure
            bar = figure.scene._tool_bar

        if show is None:
            if hasattr(bar, 'isVisible'):
                show = not bar.isVisble()
            elif hasattr(bar, 'Shown'):
                show = not bar.Shown()

        if hasattr(bar, 'setVisible'):
            bar.setVisible(show)
        elif hasattr(bar, 'Show'):
            bar.Show(show)


# show surface
# ---------------------------------------------------------------------------------------------------------
class Visualization(HasTraits):

    scene = Instance(MlabSceneModel, ())

    view = View(Item("scene", height=400, width=400,
                     editor=SceneEditor(scene_class=MayaviScene), show_label=False),
                resizable=True)


class SurfaceView(QWidget):

    # Signals
    seed_picked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(SurfaceView, self).__init__(parent)

        # initialize GUI
        screen_geo = QDesktopWidget().screenGeometry()
        self.setMinimumSize(screen_geo.width()/3, screen_geo.height()*2/3)
        self.setBackgroundRole(QPalette.Dark)

        # get mayavi scene
        # The edit_traits call will generate the widget to embed.
        self.visualization = Visualization()
        surf_viz_widget = self.visualization.edit_traits(parent=self, kind="subpanel").control
        # self.ui.setParent(self)
        figure = mlab.gcf()
        _toggle_toolbar(figure, True)

        # Initialize some fields
        self.surface_model = None
        self.painter_status = None
        self.surf = None
        self.coords = None
        self.faces = None
        self.rgba_lut = None
        self.gcf_flag = True
        self.seed_flag = False
        self.scribing_flag = False
        self.edge_list = None
        self.point_id = None
        self.old_hemi = None
        self.plot_start = None
        self.path = []
        self.cbar = None
        self._show_cbar = False

        hlayout = QHBoxLayout()
        hlayout.addWidget(surf_viz_widget)
        self.setLayout(hlayout)

    def _show_surface(self):
        """
        render the overlays
        """

        hemis = self.surface_model.get_data()
        visible_hemis = [hemi for hemi in hemis if hemi.is_visible()]
        if self.old_hemi != visible_hemis:
            self.edge_list = None
            self.old_hemi = visible_hemis

        # clear the old surface
        if self.surf is not None:
            self.surf.remove()
            self.surf = None
        if self.cbar is not None:
            self.cbar.visible = False

        # flag
        first_hemi_flag = True

        # reset
        nn = None
        self.rgba_lut = None
        vertex_number = 0

        for hemi in visible_hemis:

            # get geometry's information
            geo = hemi.current_geometry()
            hemi_coords = geo.coords
            hemi_faces = geo.faces.copy()  # need to be amended in situ, so need copy
            hemi_nn = geo.nn

            # get the rgba_lut
            rgb_array = hemi.get_composite_rgb()
            hemi_vertex_number = rgb_array.shape[0]
            alpha_channel = np.ones((hemi_vertex_number, 1), dtype=np.uint8)*255  # Mayavi uses alpha between 0~255
            hemi_lut = np.c_[rgb_array, alpha_channel]

            if first_hemi_flag:
                first_hemi_flag = False
                self.coords = hemi_coords
                self.faces = hemi_faces
                nn = hemi_nn
                self.rgba_lut = hemi_lut
            else:
                self.coords = np.r_[self.coords, hemi_coords]
                hemi_faces += vertex_number
                self.faces = np.r_[self.faces, hemi_faces]
                nn = np.r_[nn, hemi_nn]
                self.rgba_lut = np.r_[self.rgba_lut, hemi_lut]
            vertex_number += hemi_vertex_number

        if visible_hemis:
            # generate the triangular mesh
            # self.v_id2c_id: Each index is a vertex id and element is a color LUT's row index.
            self.v_id2c_id = np.arange(vertex_number)
            if len(visible_hemis) == 1:
                hemi = visible_hemis[0]
                self.top_ol = hemi.top_visible_layer
                other_ol_visibility = [ol.is_visible() for ol in hemi.overlays if ol is not self.top_ol]
                if self.top_ol is not None and self.top_ol.is_opaque():
                    _only_top_displayed = True
                elif self.top_ol is not None and not np.any(other_ol_visibility):
                    _only_top_displayed = True
                else:
                    _only_top_displayed = False
            else:
                _only_top_displayed = False

            if _only_top_displayed and not self.top_ol.is_label() and self.top_ol.get_colormap() != 'rainbow':
                self._show_cbar = True
            else:
                self._show_cbar = False

            self.mesh = self.visualization.scene.mlab.pipeline.triangular_mesh_source(self.coords[:, 0],
                                                                                      self.coords[:, 1],
                                                                                      self.coords[:, 2],
                                                                                      self.faces)
            self.mesh.data.point_data.normals = nn
            self.mesh.data.cell_data.normals = None

            if self._show_cbar:
                # get scalars
                # limit it as FreeROI normalization style
                scalars = self.top_ol.get_current_map().copy()
                scale = 255.0
                scalars = _normalize255(scalars,
                                        (self.top_ol.get_vmin(), self.top_ol.get_vmax()),
                                        scale_length=scale)
                self.scalars = scalars*(self.top_ol.get_vmax()-self.top_ol.get_vmin())/scale + self.top_ol.get_vmin()
                # get LUT
                data = np.arange(256)
                colormap = self.top_ol.get_colormap()
                if isinstance(colormap, LabelConfig):
                    colormap = colormap.get_colormap()
                self.lut_opaque = array2qrgba(data, self.top_ol.get_alpha(), colormap)
                self.lut_opaque[:, 3] = 255
                self.lut_opaque[0, :3] = np.ones((1, 3)) * 127.5

                self.mesh.mlab_source.scalars = self.scalars
                # generate the surface
                self.surf = self.visualization.scene.mlab.pipeline.surface(self.mesh,
                                                                           vmin=self.top_ol.get_vmin(),
                                                                           vmax=self.top_ol.get_vmax())
                self.surf.module_manager.scalar_lut_manager.lut.table = self.lut_opaque

                # colorbar is only meaningful for this situation
                self.cbar = mlab.colorbar(self.surf, orientation='vertical', label_fmt='%.2f')
                self.cbar.scalar_bar_representation.position = [0.85, 0.1]  # position of bottom-left corner
                self.cbar.scalar_bar_representation.position2 = [0.13, 0.8]  # distance from the bottom-left corner

            else:
                self.mesh.mlab_source.scalars = self.v_id2c_id

                # generate the surface
                self.surf = self.visualization.scene.mlab.pipeline.surface(self.mesh)
                self.surf.module_manager.scalar_lut_manager.lut.table = self.rgba_lut
                # self.surf.module_manager.scalar_lut_manager.load_lut_from_list(self.rgba_lut/255.)  # bad speed

        # add point picker observer
        if self.gcf_flag:
            self.gcf_flag = False
            fig = mlab.gcf()
            fig.on_mouse_pick(self._picker_callback_left)
            fig.scene.picker.tolerance = 0.01
            # fig.scene.scene.interactor.add_observer('MouseMoveEvent', self._move_callback)
            fig.scene.picker.pointpicker.add_observer("EndPickEvent", self._picker_callback)

    def _picker_callback(self, picker_obj, evt):

        picker_obj = tvtk.to_tvtk(picker_obj)
        self.point_id = picker_obj.point_id
        self.surface_model.set_point_id(self.point_id)

        if self.point_id != -1:
            # for painter_status
            if self.painter_status.is_drawing_valid():
                value = self.painter_status.get_drawing_value()
                if self.painter_status.is_roi_tool():
                    roi_val = self.surface_model.data(self.surface_model.current_index(),
                                                      QtCore.Qt.UserRole + 4)
                    self.surface_model.set_vertices_value(value, roi=roi_val)
                else:
                    size = self.painter_status.get_drawing_size()
                    vertices = [self.point_id]
                    if size != 0:
                        vertices.extend(list(get_vtx_neighbor(self.point_id, self.faces, size)))
                    self.surface_model.set_vertices_value(value, vertices=vertices)

            else:
                if self.painter_status.is_roi_selection():
                    roi_val = self.surface_model.data(self.surface_model.current_index(),
                                                      QtCore.Qt.UserRole + 4)
                    self.painter_status.get_draw_settings()._update_roi(roi_val)

                self.tmp_lut = self.rgba_lut.copy()

                # plot line
                if self.scribing_flag:
                    if self.edge_list is None:
                        self.create_edge_list()
                    self._plot_line()

                # get seed
                if self.seed_flag:
                    self.seed_picked.emit()

                # plot point
                c_id = self.v_id2c_id[self.point_id]
                toggle_color(self.tmp_lut[c_id])
                if self._show_cbar:
                    self.cbar.visible = False
                    # self.surf.mlab_source.scalars = self.v_id2c_id
                    self.mesh.mlab_source.scalars = self.v_id2c_id
                    self.surf.remove()
                    self.surf = self.visualization.scene.mlab.pipeline.surface(self.mesh)
                self.surf.module_manager.scalar_lut_manager.lut.table = self.tmp_lut
        elif self._show_cbar:
            self.mesh.mlab_source.scalars = self.scalars
            self.surf.remove()
            self.surf = self.visualization.scene.mlab.pipeline.surface(self.mesh,
                                                                       vmin=self.top_ol.get_vmin(),
                                                                       vmax=self.top_ol.get_vmax())
            self.surf.module_manager.scalar_lut_manager.lut.table = self.lut_opaque
            # FIXME In this case, I can't use 'mlab.colorbar' to recreate the color bar and
            # it is only able to be turned on its visibility if I want to show it again. But,
            # it will appear with strange size, and I can't adjust it by using 'position' and 'position2'.
            self.cbar.visible = True

        azimuth, elevation, distance, focalpoint = mlab.view()
        roll = mlab.roll()
        self.surface_model.camera_to_edit(azimuth, elevation, distance, focalpoint, roll)

    def _picker_callback_left(self, picker_obj):
        pass

    def _create_connections(self):
        self.surface_model.repaint_surface.connect(self._show_surface)
        self.connect(self.surface_model, QtCore.SIGNAL("camera_to_show"), self.set_camera)

    def _plot_line(self):
        if self.plot_start is None:
            self.plot_start = self.point_id
            self.path.append(self.plot_start)
            self._origin = self.point_id  # the origin of this plot
        else:
            if self.point_id in self.edge_list[self._origin]:
                # Make the line's head and tail more easily closed
                self.point_id = self._origin

            new_path = bfs(self.edge_list, self.plot_start, self.point_id,
                           deep_limit=50)
            if new_path:
                self.plot_start = self.point_id
                new_path.pop(0)
                self.path.extend(new_path)
            else:
                QMessageBox.warning(
                    self,
                    'Warning',
                    'There is no line linking the start and end vertices.\n'
                    'Or the line is too long.\nPlease select the end vertex again.',
                    QMessageBox.Yes
                )

            for v_id in self.path:
                c_id = self.v_id2c_id[v_id]
                toggle_color(self.tmp_lut[c_id])

    # user-oriented methods
    # -----------------------------------------------------------------
    def set_model(self, surface_model):
        if isinstance(surface_model, TreeModel):
            self.surface_model = surface_model
            self._create_connections()
            self._show_surface()
        else:
            raise ValueError("The model must be the instance of the TreeModel!")

    def set_painter_status(self, painter_status):
        self.painter_status = painter_status

    def create_edge_list(self):
        self.edge_list = get_n_ring_neighbor(self.faces)

    def get_coords(self):
        return self.coords

    def get_faces(self):
        return self.faces

    def set_camera(self, azimuth, elevation, distance, focalpoint, roll):
        mlab.view(azimuth, elevation, distance, focalpoint)
        mlab.roll(roll)
