import bpy
from pathlib import Path


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


def select(obj, active=False):
    obj.select_set(True)
    if active:
        bpy.context.view_layer.objects.active = obj


def make_obj_smooth(obj, iterations=10, factor=2):
    """Adds smoothing modifier in Blender"""

    # Add smoothing modifier
    smoothing = obj.modifiers.new(name="Smooth", type="SMOOTH")
    smoothing.iterations = iterations
    smoothing.factor = factor

    # Recalculate normals
    bpy.ops.object.select_all(action="DESELECT")
    select(obj, True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()

    # Since there are no normals in the vertices, add double-sided rendering for faces to fix artifacts
    # Update: not needed since normals are now calculated
    # bpy.data.meshes[obj.name].show_double_sided = True

    bpy.ops.object.shade_smooth()


if __name__ == "__main__":

    filePath = Path(r'E:\conda_projects\Airway\data\CY_14210425\S1_modelForSkeVoxel.obj')
    
    remove_all_data()
    remove_custom_properties()
    
    bpy.ops.import_scene.obj(filepath=str(filePath))
    model = bpy.data.objects[filePath.stem]
    make_obj_smooth(model) 
