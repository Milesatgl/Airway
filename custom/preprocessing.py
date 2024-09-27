import trimesh, sys
from trimesh import voxel
import numpy as np
from pathlib import Path
from utils import getOutputPath
# stlPath = sys.argv[1]

def stl2Model(stlPath: Path, closing=False, exportObj=True, voxelScale=1.0):
    '''Voxelize stl model and transform pose'''

    exampleDataShape = (582, 208, 319)

    stlMesh = trimesh.load_mesh(stlPath)

    stlSize: np.ndarray = stlMesh.extents  # type: ignore
    scale = round(max(exampleDataShape) / stlSize.max(), 1)
    print(f'Standard shape is: {exampleDataShape}, current: {stlSize}, scale: {scale}')
    stlMesh.apply_scale(scale) # type: ignore

    stlVoxel = voxel.creation.voxelize(mesh=stlMesh, pitch=1)
    stlVoxel.fill()

    if closing:
        closedVoxel = trimesh.voxel.morphology.binary_closing(stlVoxel.matrix) # type: ignore
        stlVoxel = voxel.VoxelGrid(closedVoxel, transform=stlVoxel.transform)
    
    # stlVoxel.apply_scale(voxelScale)

    if exportObj:
        voxelMesh = stlVoxel.as_boxes()
        voxelMesh.export(getOutputPath(stlPath, f'_voxel_S({scale}).obj', 'S0_'))

    stlArr = stlVoxel.matrix # Matrix of bool
    # To target pose
    stlArr = np.transpose(stlArr[:, :, ::-1], (2, 1, 0))
    # Pad each dimention
    stlArr = np.pad(
        stlArr, ((10, 10), (10, 10), (10, 10)), mode='constant',
        constant_values=0
    )
    # outPath = Path(stlPath).parent / 'model.npz'
    # np.savez_compressed(str(outPath), stlArr)
    return stlArr


def print_model_description(model: np.ndarray):
    total_sum = np.sum(model)
    print(f"Total sum: {total_sum:,}")
    print(f"Total pixels in model: {np.prod(np.array(model.shape)):,}")
    return total_sum


def removeAll0Layers(voxelModel: str | np.ndarray):
    if isinstance(voxelModel, str):
        model = np.load(voxelModel)['arr_0']
    elif isinstance(voxelModel, np.ndarray):
        model = voxelModel
    else:
        return
    
    model = model.astype(np.uint8)
    unique, counts = np.unique(model, return_counts=True)

    assert len(unique) != 1, f"It looks like the the model only contains {unique[0]}s, aborting!"
    print("{} images loaded".format(len(model)))
    print("Printing sum as validation as only 0-layers are being removed the sum should not change.")
    print("Before reduction:")
    old_total_sum = print_model_description(model)

    # Axis description:
    #      0: top to bottom
    #      1: front to back
    #      2: left to right

    print("\nReducing model: ", end="")
    print(model.shape, end=" ")

    for axis in [0, 1, 2]:
        sums = np.sum(np.sum(model, axis=axis), axis=(axis + 1) % 2)

        # Track all =0 layers from front from that axis
        remove_front_index = 0
        while sums[remove_front_index] == 0:
            remove_front_index += 1

        # Track all =0 layers from back from that axis
        remove_back_index = len(sums) - 1
        while sums[remove_back_index] == 0:
            remove_back_index -= 1

        # Remove those layers
        model = np.delete(
            model, list(range(remove_front_index - 1)) + list(range(remove_back_index + 2, len(sums))), axis=(axis + 1) % 3
        )
        validation_sums = np.sum(np.sum(model, axis=axis), axis=(axis + 1) % 2)
        print(" -> ", model.shape, end=" ")

    assert all(a > 2 for a in model.shape), f"Model is empty! shape={model.shape}"

    print("\n\nAfter reduction:")
    curr_total_sum = print_model_description(model)

    if curr_total_sum == old_total_sum:
        # np.savez_compressed(reducedModelDir / "reduced_model", model)
        return model
    else:
        raise Exception("It seems like the script removed actual data from the model; this should not happen!")
    

def preprocessingStl(stlPath: Path, saveRes = True, voxelScale=1.0) -> np.ndarray:
    voxelModel = stl2Model(stlPath, closing=True, exportObj=saveRes, voxelScale=voxelScale)
    if saveRes:
        np.savez_compressed(
            str(getOutputPath(stlPath, '_voxel', 'S0_')),
            voxelModel
        )

    preprocessedModel = removeAll0Layers(voxelModel=voxelModel)

    if saveRes:
        np.savez_compressed(str(getOutputPath(stlPath, '_preprocessed', 'S0_')), preprocessedModel)

    return preprocessedModel


if __name__ == "__main__":
    '''
    preprocessing -> bfsDistance -> createTree -> composeTree -> postProcessingTree -> splitClassification
    '''
    # stlPath = Path(r'E:\conda_projects\Airway\data\CY_14210425\Airway.stl')
    import sys
    stlPath = Path(sys.argv[1]).joinpath(f'Airway.stl')
    # voxelScale = float(sys.argv[2])
    _ = preprocessingStl(stlPath, True)


# "D:\mambaforge\envs\skeletonA\python.exe" E:\conda_projects\Airway\custom\preprocessing.py E:\conda_projects\Airway\data\HY_26577711 1.0
    