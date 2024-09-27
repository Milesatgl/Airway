import numpy as np
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction
from tvtk.util.ctf import PiecewiseFunction
import moviepy.editor as mpy
import monai.transforms as mt
# import time
# import multiprocessing
from pyface.api import GUI
from pathlib import Path
from typing import List, Tuple
import matplotlib.colors as mcolors
# import sys
# sys.path.append(r'E:\conda_projects\ctSegTest')
# from custom.multiScene import MyDialog2
# from copy import deepcopy
# from pyface.i_gui import IGUI
# pip install mayavi moviepy

# mayavi数据可视化 -----------------------------------------------------------------

def animateMayavi(gifName: str):
    '''还有点问题'''
    def make_frame(t):
        # camera angle
        mlab.view(azimuth=360 * t / duration, elevation=-70, distance='auto', focalpoint='auto')
        return mlab.screenshot(antialiased=True)

    duration = 5
    animation = mpy.VideoClip(make_frame, duration=duration)

    animation.write_gif(gifName, fps=20)
    mlab.close()


# def test_mlab_show():
#     """Test mlab.show()"""
#     run_mlab_examples()
#     # Automatically close window in 100 msecs.
#     GUI.invoke_after(100, mlab.close)
#     mlab.show()

def get1labelVol(label: np.ndarray, opt=0.5, color=(1, 0, 0), scene=None):

    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(label)) if scene is None else \
        mlab.pipeline.volume(mlab.pipeline.scalar_field(label), figure=scene.mayavi_scene)
    
    otf = PiecewiseFunction()
    ctf = ColorTransferFunction()
    ctf.add_rgb_point(1, *color)
    vol._volume_property.set_color(ctf)
    vol._ctf = ctf
    vol.update_ctf = True
    otf.add_point(0, 0.0)
    otf.add_point(1, opt)
    vol._otf = otf
    vol._volume_property.set_scalar_opacity(otf)
    return vol


def showOrAnimate(
        gifName: str, duration=10, name: str | List[str] = 'label',
        nameColor: Tuple[float, float, float] | List[Tuple[float, float, float]] = (0, 0, 0),
):
    '''提供gifName（存储路径）则保存为gif，否则直接显示'''
    if gifName:
        animateMayavi(gifName)
    else:
        # 当show时间超过10s时，自动关闭
        if isinstance(name, str):
            name = [name]
        if isinstance(nameColor, tuple):
            nameColor = [nameColor]
        assert len(name) == len(nameColor), f'len(name) != len(nameColor), {len(name)} != {len(nameColor)}'
        # set text interval and text width accoding to the number of labels
        # name = name[::-1]
        # nameColor = nameColor[::-1]
        textWidth = 1/(len(name)+1) if len(name) > 1 else 0.5
        interval = textWidth/(len(name)+1) if len(name) > 1 else 0.1
        GUI.invoke_after(1000*duration, mlab.close, all=True) if duration else None

        for i, (n, c) in enumerate(zip(name[::-1], nameColor[::-1])):
            mlab.text((i+1)*interval+i*textWidth, 0.0, n, width=textWidth, color=c)

        # mlab.text(0.0, 0.0, name, width=0.5, color=(0, 0, 0)) if name else None
        mlab.outline(color=(0, 1, 0))
        mlab.show()


def show1label(label: np.ndarray, figSize=(800, 800), gifName: str = None, duration=10, name='test'):
    '''提供gifName（存储路径）则保存为gif，否则直接显示'''
    if gifName:
        assert '.gif' in gifName, f'gifName: {gifName} is not a gif file'

    fig = mlab.figure(bgcolor=(1, 1, 1), size=figSize)
    vol = get1labelVol(label)

    showOrAnimate(gifName, duration, name)


def show2labelIn1Fig(
        label1: np.ndarray, label2: np.ndarray,
        gifName: str = None, duration=10, name1='label1', name2='label2',
        opt1=0.5, opt2=0.5, color1=(1, 0, 0), color2=(0, 1, 0)
):
    if gifName:
        assert '.gif' in gifName, f'gifName: {gifName} is not a gif file'

    fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    vol1 = get1labelVol(label1, opt=opt1, color=color1)
    # vol1Zeros = np.zeros_like(label1)
    # label2 = np.concatenate([vol1Zeros, label2], axis=0)
    vol2 = get1labelVol(label2, opt=opt2, color=color2)

    showOrAnimate(gifName, duration, [name1, name2], [color1, color2])


# def show2labelIn1Fig2(
#         label1: np.ndarray, label2: np.ndarray, name1='label1', name2='label2',
#         opt1=0.5, opt2=0.5, color1=(1, 0, 0), color2=(0, 1, 0)
# ):

#     dialog = MyDialog2(name1=name1, name2=name2)
#     vol1 = get1labelVol(label1, opt=opt1, color=color1, scene=dialog.scene1)
#     vol2 = get1labelVol(label2, opt=opt2, color=color2, scene=dialog.scene2)
#     dialog.configure_traits()
    
    # showOrAnimate(gifName, duration, [name1, name2], [color1, color2])

# 
def plotConnectedLabel(connected: np.ndarray):

    otf = PiecewiseFunction()
    ctf = ColorTransferFunction()

    labels, num= np.unique(connected, return_counts=True)

    # sort label by label count
    labelInfos = sorted(zip(labels, num), key=lambda x: x[1], reverse=True)
    # for label, count in labelInfos:
    #     print('label', label, '; count', count)

    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(connected))

    for i, (label, _ ) in enumerate(labelInfos):
        if label == 0:
            continue
        # set different colors for different labels
        # print('changing color for', label)
        if i == 1:
            # 主干就用红色
            ctf.add_rgb_point(label, 1, 0, 0)
        else:
            ctf.add_rgb_point(label, *mcolors.hsv_to_rgb([label/len(labels), 1, 1]))

    vol._volume_property.set_color(ctf)
    vol._ctf = ctf
    vol.update_ctf = True

    for i, (label, count) in enumerate(labelInfos):
        if label == 0:
            otf.add_point(label, 0.0)
        else:
            # set different opacity for different labels based on label count
            if i == 1:
                otf.add_point(label, 0.2)
            else:
                otf.add_point(label, 1.0)

    vol._otf = otf
    vol._volume_property.set_scalar_opacity(otf)
    mlab.colorbar()
    mlab.show()


if __name__ == '__main__':
    import time
    targetNii = r'E:\conda_projects\ctSegTest\data\nnUnet\nnUNet_raw\Dataset011_ATMLungAirWay\lungMaskTr\ATM_059_0000_lungMask.nii.gz'
    img = mt.LoadImage(image_only=True)(targetNii).numpy()
    img = img.astype(float)
    gifName = None
    start = time.time()
    # show1label(img, gifName=gifName, duration=0, name=Path(targetNii).name)
    show2labelIn1Fig(img, img, gifName=gifName, duration=0, name1='label1', name2='label2', opt1=1.0, opt2=0.1)
    # show2labelIn1Fig2(img, img)
    print(f'time: {time.time()-start:.2f}s')