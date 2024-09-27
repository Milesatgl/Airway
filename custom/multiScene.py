import numpy as np

from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group

from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
            SceneEditor
from tvtk.util.ctf import ColorTransferFunction
from tvtk.util.ctf import PiecewiseFunction
from pyface.api import GUI
# import monai.transforms as mt
from copy import deepcopy
from mayavi.core.api import Engine

class MyDialog(HasTraits):

    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    button1 = Button('Redraw')
    button2 = Button('Redraw')

    @on_trait_change('button1')
    def redraw_scene1(self):
        self.redraw_scene(self.scene1)

    @on_trait_change('button2')
    def redraw_scene2(self):
        self.redraw_scene(self.scene2)

    def redraw_scene(self, scene):
        # Notice how each mlab call points explicitly to the figure it
        # applies to.
        mlab.clf(figure=scene.mayavi_scene)
        x, y, z, s = np.random.random((4, 100))
        mlab.points3d(x, y, z, s, figure=scene.mayavi_scene)

    # The layout of the dialog created
    view = View(HSplit(
                  Group(
                       Item('scene1',
                            editor=SceneEditor(), height=250,
                            width=300),
                       'button1',
                       show_labels=False,
                  ),
                  Group(
                       Item('scene2',
                            editor=SceneEditor(), height=250,
                            width=300, show_label=False),
                       'button2',
                       show_labels=False,
                  ),
                ),
                resizable=True,
                )


# m = MyDialog()
# m.configure_traits()


class MyDialog2(HasTraits):
    def __init__(self, *args, **kwargs):
        # super().__init__()
        self.volData1 = kwargs['volData1']
        self.volData2 = kwargs['volData2']
        self.opt1 = kwargs.get('opt1', 0.5)
        self.opt2 = kwargs.get('opt2', 0.5)
        self.name1 = kwargs.get('name1', 'vol1')
        self.name2 = kwargs.get('name2', 'vol2')
        self.color1 = kwargs.get('color1', (1, 0, 0))
        self.color2 = kwargs.get('color2', (0, 1, 0))

        # vol1 = scene1.mlab.pipeline.volume(mlab.pipeline.scalar_field(self.volData1), figure=scene1.mayavi_scene)
        # vol2 = scene2.mlab.pipeline.volume(mlab.pipeline.scalar_field(self.volData2), figure=scene2.mayavi_scene)

    @on_trait_change('scene1.activated')
    def populate_scene1(self):
        # self.scene1.mlab.pipeline.volume(self.scene1.mlab.pipeline.scalar_field(self.volData1))
        mlab.test_surf()

    @on_trait_change('scene2.activated')
    def populate_scene2(self):
        # self.scene2.mlab.pipeline.volume(self.scene2.mlab.pipeline.scalar_field(self.volData2))
        mlab.test_surf()

    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    
    view = View(HSplit(
            Group(
                Item('scene1',
                        editor=SceneEditor(scene_class=MayaviScene), height=250,
                        width=300),
                show_labels=False,
            ),
            Group(
                Item('scene2',
                        editor=SceneEditor(scene_class=MayaviScene), height=250,
                        width=300),
                show_labels=False,
            ),
            ),
            resizable=True,
            )
    

class MyDialog3(HasTraits):
    def __init__(self, *args, **kwargs):
        super(MyDialog3, self).__init__()
        # self.volData1 = kwargs['volData1']
        # self.volData2 = kwargs['volData2']
        # self.opt1 = kwargs.get('opt1', 0.5)
        # self.opt2 = kwargs.get('opt2', 0.5)
        # self.name1 = kwargs.get('name1', 'vol1')
        # self.name2 = kwargs.get('name2', 'vol2')
        # self.color1 = kwargs.get('color1', (1, 0, 0))
        # self.color2 = kwargs.get('color2', (0, 1, 0))

        self.scene1 = Instance(MlabSceneModel, ())
        self.scene2 = Instance(MlabSceneModel, ())

    button1 = Button('Redraw')
    button2 = Button('Redraw')

    @on_trait_change('scene1.activated')
    def redraw_scene1(self):
        self.redraw_scene(self.scene1)

    @on_trait_change('scene2.activated')
    def redraw_scene2(self):
        self.redraw_scene(self.scene2)

    def redraw_scene(self, scene):
        # mlab.pipeline.volume(mlab.pipeline.scalar_field(self.volData2, figure=scene.mayavi_scene), figure=scene.mayavi_scene)
        # mlab.test_surf()
        x, y, z, s = np.random.random((4, 100))
        mlab.points3d(x, y, z, s, figure=scene.mayavi_scene)

    view = View(HSplit(
                Group(
                    Item('scene1',
                            editor=SceneEditor(), height=250,
                            width=300),
                    show_labels=False,
                ),
                Group(
                    Item('scene2',
                            editor=SceneEditor(), height=250,
                            width=300, show_label=False),
                    show_labels=False,
                ),
                ),
                resizable=True,
                )
    
    # def __init__(self, *args, **kwargs):
    #     # super().__init__()
    #     self.volData1 = kwargs['volData1']
    #     self.volData2 = kwargs['volData2']
    #     self.opt1 = kwargs.get('opt1', 0.5)
    #     self.opt2 = kwargs.get('opt2', 0.5)
    #     self.name1 = kwargs.get('name1', 'vol1')
    #     self.name2 = kwargs.get('name2', 'vol2')
    #     self.color1 = kwargs.get('color1', (1, 0, 0))
    #     self.color2 = kwargs.get('color2', (0, 1, 0))

class MyDialog4(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data1 = kwargs['data1']
        self.data2 = kwargs['data2']

    @on_trait_change('scene1.activated')
    def redraw_scene1(self):
        # mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data1), figure=self.scene1.mayavi_scene)
        # self.scene1.isometric_view()
        # self.scene1.show_axes = True
        # mlab.axes(figure=self.scene1.mayavi_scene)
        # mlab.view(
        #     azimuth=100, elevation=100, distance='auto',
        #     figure=self.scene1.mayavi_scene,
        #     focalpoint='auto'
        # )
        self.scene1.mlab.pipeline.volume(self.scene1.mlab.pipeline.scalar_field(self.data1))
        # self.scene1.mayavi_scene.isometric_view()
        # scene = self.scene1.mayavi_scene.scene
        # if hasattr(scene, 'isometric_view'):
        #     print(f'[INFO] fond isometric_view')
        #     scene.isometric_view()
        # else:
        #     # Not every viewer might implement this method
        #     # self.scene1.view(40, 50)
        #     print(help(scene))


    @on_trait_change('scene2.activated')
    def redraw_scene2(self):
        mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data2), figure=self.scene2.mayavi_scene)

    view = View(HSplit(
                  Group(
                       Item('scene1',
                            editor=SceneEditor(), height=250,
                            width=300),
                    #    'button1',
                       show_labels=False,
                  ),
                  Group(
                       Item('scene2',
                            editor=SceneEditor(), height=250,
                            width=300, show_label=False),
                    #    'button2',
                       show_labels=False,
                  ),
                ),
                resizable=True,
                )


class MyDialog5(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    scene3 = Instance(MlabSceneModel, ())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data1 = kwargs['data1']
        self.data2 = kwargs['data2']
        self.data3 = kwargs['data3']

    @on_trait_change('scene1.activated')
    def redraw_scene1(self):
        self.scene1.mlab.pipeline.volume(self.scene1.mlab.pipeline.scalar_field(self.data1))


    @on_trait_change('scene2.activated')
    def redraw_scene2(self):
        self.scene2.mlab.pipeline.volume(self.scene2.mlab.pipeline.scalar_field(self.data2))

    @on_trait_change('scene3.activated')
    def redraw_scene3(self):
        self.scene3.mlab.pipeline.volume(self.scene3.mlab.pipeline.scalar_field(self.data3))

    view = View(HSplit(
                  Group(
                       Item('scene1',
                            editor=SceneEditor(), height=300,
                            width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene2',
                            editor=SceneEditor(), height=300,
                            width=300, show_label=False),
                       show_labels=False,
                  ),
                  Group(
                          Item(
                              'scene3',
                                editor=SceneEditor(), height=300,
                                width=300, show_label=False
                          
                          ),
                        show_labels=False,
                  )
                ),
                resizable=True,
                )

if __name__ == '__main__':
    data1Path = r'E:\conda_projects\ctSegTest\data\nnUnet\nnUNet_raw\Dataset011_ATMLungAirWay\labelsTr\ATM_118.nii.gz'
    data2Path = r'E:\conda_projects\ctSegTest\data\nnUnet\nnUNet_raw\Dataset011_ATMLungAirWay\predCpu_segTest\ATM_118_0000_seg.nii.gz'
    data3Path = r'E:\conda_projects\ctSegTest\data\nnUnet\nnUNet_raw\Dataset011_ATMLungAirWay\lungMaskTrRefine\ATM_118_0000_lungMask_Refined.nii.gz'
    data4Path = r'E:\conda_projects\ctSegTest\data\nnUnet\nnUNet_raw\Dataset011_ATMLungAirWay\lungMaskTr\ATM_118_0000_lungMask.nii.gz'
    data1 = mt.LoadImage(image_only=True)(data4Path).numpy().astype(float)
    data2 = mt.LoadImage(image_only=True)(data2Path).numpy().astype(float)
    MyDialog4(data1=data1, data2=data2).configure_traits()

