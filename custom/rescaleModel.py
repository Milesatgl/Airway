import trimesh
import sys, re
from pathlib import Path
from trimesh.exchange.obj import export_obj

projDir = Path(sys.argv[1])

pattern = r'S0_Airway_voxel_S\((.*?)\)\.obj'

scale = None

for objPath in projDir.glob('*.obj'):
    match = re.match(pattern, objPath.name)
    if match:
        numberStr = match.group(1)
        try:
            scale = float(numberStr)
        except ValueError:
            print(f'Cannot convert {numberStr} to float')
            continue
        break

if scale is None:
    print(f'Unable to find scale, exiting')
    sys.exit(-1)


voxelObj = projDir.joinpath(f'S6_S1_skeletonVoxel_label.obj')
assert voxelObj.exists(), f'Unable to find target file: {voxelObj}'

voxelMesh = trimesh.load_mesh(voxelObj)
voxelMesh.apply_scale(1.0/scale) # type: ignore
# targetMeshName = 'S7_skeletonVoxel_label_rescale'
voxelMesh.export( # type: ignore
    projDir.joinpath(f'S7_skeletonVoxel_label_rescale.obj'),
    file_type='obj'
)

modelObj = projDir.joinpath(f'S1_modelForSkeVoxel.obj')
modelMesh = trimesh.load_mesh(modelObj)
modelMesh.apply_scale(1.0/scale) # type: ignore
modelMesh.export( # type: ignore
    projDir.joinpath(f'S7_modelForSkeVoxel_rescale.obj'),
    file_type='obj'
)

# export_obj(targetMesh, include_color=False, include_texture=False, write_texture=False, mtl_name=)

# "D:\mambaforge\envs\skeletonA\python.exe" E:\conda_projects\Airway\custom\rescaleModel.py E:\conda_projects\Airway\data\HY_26577711