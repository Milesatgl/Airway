import bpy # type: ignore
import math
from mathutils import Vector # type: ignore
from pathlib import Path
import networkx as nx
import numpy as np
from typing import Dict, Any, List


def remove_all_data(remove_camera: bool = True):
    """ Remove all data blocks except opened scripts, the default scene and the camera.
    :param remove_camera: If True, also the default camera is removed.
    """
    # Go through all attributes of bpy.data
    for collection in dir(bpy.data):
        data_structure = getattr(bpy.data, collection)
        # Check that it is a data collection
        if isinstance(data_structure, bpy.types.bpy_prop_collection) and hasattr(data_structure, "remove") \
                and collection not in ["texts"]:
            # Go over all entities in that collection
            for block in data_structure:
                # Skip the default scene
                if isinstance(block, bpy.types.Scene) and block.name == "Scene":
                    continue
                # If desired, skip camera
                if not remove_camera and isinstance(block, (bpy.types.Object, bpy.types.Camera)) \
                        and block.name == "Camera":
                    continue
                data_structure.remove(block)
                
def remove_custom_properties():
    """ Remove all custom properties registered at global entities like the scene. """
    for key in bpy.context.scene.keys():
        del bpy.context.scene[key]


def extractLabel(tree: nx.Graph):
    label: Dict[str, Dict[str, Any]] = {}
    allNodes = tree.nodes
    for nodeId in list(allNodes):
        nodeInfo = {
            'coords': (allNodes[nodeId]['x'], allNodes[nodeId]['y'], allNodes[nodeId]['z']),
            'name': allNodes[nodeId]['split_classification']
        }
        label.update({nodeId: nodeInfo})
    
    return label


def addLabel(targetObj, labelInfo: Dict[str, Dict[str, Any]]):

    textObjs: List[str] = []

    # mesh = targetObj.data
    for k, v in labelInfo.items():
        text = f"{k}_{v['name']}"
        if v['name'][0] == 'c':
            continue

        location = targetObj.matrix_world @ Vector(v['coords']) + Vector((1, -1, -2))

        # if k in ['0', '1', '2']:
        #     print(f"\n\nNODE0 ORIGINAL_LOC: {Vector(v['coords'])}\nTARGET_MATRIX: {targetObj.matrix_world}\nTRANSFORMED_LOC: {location}\n\n")

        bpy.ops.object.text_add(location=location)
        textObj = bpy.context.active_object
        textObj.data.body = text
        textObj.scale *= 3
        textObj.rotation_euler[2] = math.pi / 2
        textObj.rotation_euler[1] = math.pi
        textObj.name = text
        bpy.context.view_layer.update()
        textObjs.append(textObj.name)

    return textObjs


def joinObjects(objNames: List[str]):
    bpy.ops.object.select_all(action='DESELECT')
    for objName in objNames:
        bpy.data.objects[objName].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[objNames[0]]
    bpy.ops.object.join()


def initialzeTarget(objPath: str, setOriginToCenter = False, orginalModelPath: str | None = None):
    bpy.ops.import_scene.obj(filepath=objPath)
    targetObj = bpy.context.selected_objects[0]

    # centroid = Vector((0, 0, 0))
    if setOriginToCenter:
        vertices = [targetObj.matrix_world @ vertex.co for vertex in targetObj.data.vertices]
        centroid = Vector(np.mean(vertices, axis=0))
        targetObj.location -= centroid
        # bpy.ops.object.origin_set(
        #     type='ORIGIN_CENTER_OF_MASS',
        #     center='BOUNDS'
        # )
        # targetObj.location = Vector((0, 0, 0))

        if orginalModelPath is not None:
            bpy.ops.import_scene.obj(filepath=orginalModelPath)
            originModelObj = bpy.data.objects[Path(orginalModelPath).stem]
            print(f'Orginal model name is {originModelObj.name}')
            originModelObj.location -= centroid

    bpy.context.view_layer.update()

    return targetObj


if __name__ == "__main__":

    import os, sys
    try:
        projDir = os.environ['projDir']
    except KeyError:
        print(f'Unable to find projDir in environment, exiting')
        sys.exit(-1)

    remove_all_data()
    remove_custom_properties()

    # skePath = Path(r'E:\conda_projects\Airway\data\CY_14210425\S1_skeletonVoxel.obj')

    skePath = Path(projDir) / 'S1_skeletonVoxel.obj'

    airTreeModelPath = skePath.parent / 'S1_modelForSkeVoxel.obj'
    classifiedTreePath = skePath.parent / r'S5_S4_postProcessingTree_classified.graphml'

    addModelToo = False
    setOriginToCenter = False

    tree = nx.read_graphml(classifiedTreePath)
    labelInfo = extractLabel(tree=tree)

    targetObj = initialzeTarget(
        str(skePath), setOriginToCenter=setOriginToCenter,
        orginalModelPath=str(airTreeModelPath) if addModelToo else None
    )

    textObjs = addLabel(
        targetObj=targetObj,
        labelInfo=labelInfo
    )
    allObjs: List[str] = [targetObj.name] + textObjs
    joinObjects(allObjs)

    outPath = skePath.parent / f'S6_{skePath.stem}_label.obj'
    bpy.ops.export_scene.obj(filepath=str(outPath))

    bpy.ops.file.pack_all()
    outBlenderPath = outPath.parent / 'final.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(outBlenderPath), compress=True, copy=True)
    
# set projDir=E:\conda_projects\Airway\data\HY_26577711
# "D:\blender338\blender.exe" -b -P E:\conda_projects\Airway\custom\labelSkeleton.py