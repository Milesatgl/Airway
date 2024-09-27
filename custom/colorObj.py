from collections import defaultdict
from pathlib import Path
from typing import List, Set, Tuple
from typing import Dict

import numpy as np
from skimage.morphology import skeletonize
from createColorMask import color_hex_to_floats


def normalize(vertices: np.ndarray, reference_shape: Tuple[int, ...], rot_mat: np.ndarray | None = None):
    # Shift to middle of the space
    vertices -= np.array(reference_shape) / 2
    # Scale to [-10..10]
    vertices *= 20 / np.max(reference_shape)
    # If available: transform
    # Note: since this is applied afterwards, points can be out of [-10..10]
    if rot_mat is not None:
        vertices = vertices @ np.transpose(rot_mat)
    return vertices


def generate_obj(
    outputDataPath: Path,
    accepted_types: Set[int],
    model: np.ndarray,
    color_mask: np.ndarray | None = None,
    color_to_rgb_tuple: Dict[int, Tuple[float, float, float]] = {},
    rot_mat: np.ndarray | None = None,
    num_decimal_digits: int = 2,
):
    """Saves a .obj obj_file given the model, the accepted types and a name

    outputDataPath is a pathlib Path, this is the full path the obj_file will be saved as

    accepted_types is a list or set which types should be looked for, other
    types will be ignored. If empty set then everything except for 0 will be
    accepted

    model is the 3D numpy array model of the lung or whatever object you
    want to convert to 3D

    color_to_rgb_tuple is a Dict which maps the color id used in color mask to a tuple
    of rgb values. This color will be used to color the object with that color.
    (e.g. {1: (1, 0.4, 0.5)})

    color_mask is a model with the same shape as model, but its numbers represent
    groups of colors/materials which will be added by this script

    rot_mat is a rotation matrix. Each point p=(x,y,z) is rotated by rot_mat @ p or left unchanged if None
    """
    occurrences = np.unique(model)
    if not all(t in occurrences for t in accepted_types):
        return

    outputDataPath = Path(outputDataPath)

    print(f"Generating {outputDataPath} with accepted types of {accepted_types}")

    vertices = {}
    faces: Dict[int, List[List[int]]] = defaultdict(list)

    model = np.pad(np.copy(model), 1)

    if color_mask is not None:
        color_mask = np.pad(color_mask, 1)
    if accepted_types:
        for remove in set(np.unique(model)) - accepted_types - {0}:
            model[model == remove] = 0

    index = 1
    # Iterate over each axis and pos/neg directions, then roll the model over, afterwards subtracting these.
    # This causes there to be only -1 and 1 values where there is air, meaning there a face should be added.
    # Though we only look at the 1 values, since these are actually in the model (the others are outside, which means
    # their color map will be wrong)
    for axis in range(3):
        for pos_or_neg in [-1, 1]:
            diff = np.roll(model, -pos_or_neg, axis=axis)
            model_diff = np.where(model > diff)
            coords = []
            for d1, d2 in [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]:
                coords.append(list(map(lambda t: t.astype(float), np.copy(model_diff))))
                coords[-1][axis] += pos_or_neg / 2
                coords[-1][(axis + 1) % 3] += d1
                coords[-1][(axis + 2) % 3] += d2

            # Example shape after transpose: (11523, 4, 3) - face_coords is an array of faces,
            # each face is a list of 4 points with 3 coordinates.
            faces_coords = np.transpose(coords, axes=(2, 0, 1))
            vertex_coords = np.transpose(model_diff)
            for vertex_coord, face_coords in zip(vertex_coords, faces_coords):
                curr_face = []
                for face_coord in map(tuple, face_coords):
                    if face_coord not in vertices:
                        vertices[face_coord] = index
                        index += 1
                    curr_face.append(vertices[face_coord])
                material = color_mask[tuple(vertex_coord)] if color_mask is not None else 0
                faces[material].append(curr_face) # type: ignore
                # assert len(faces[material][-1]) == 4, f"ERROR: Wrong number of points on face {faces[material][-1]}"

    print(f"Vertex count : {len(vertices):,}")
    print(f"Face count : {sum(map(len, faces.values())):,}")

    # make to numpy for easier usage later
    vertices = np.array([np.array(v) for v in vertices])
    vertices = normalize(vertices, model.shape, rot_mat=rot_mat)

    # Write vertices and faces to obj_file
    material_path = outputDataPath.with_suffix(".mtl")
    with open(material_path, "w") as mat_file:
        import random

        random.seed(outputDataPath.parent.name)

        def ran():
            return random.uniform(0, 1)

        for material in faces:
            mat_file.write(f"newmtl mat{material}\n")
            mat_file.write("Ns 96.078431\n")
            mat_file.write("Ka 1.000000 1.000000 1.000000\n")
            rgb = color_to_rgb_tuple[material] if material in color_to_rgb_tuple else (ran(), ran(), ran())
            mat_file.write(f"Kd {' '.join(map(str, rgb))}\n")
            mat_file.write("Ks 0.500000 0.500000 0.500000\n")
            mat_file.write("Ke 0.000000 0.000000 0.000000\n")
            mat_file.write("Ni 1.000000\n")
            mat_file.write("d 1.000000\n")
            mat_file.write("illum 2\n\n")
    with open(outputDataPath, "w") as obj_file:
        obj_file.write("# .obj generated by Airway")
        obj_file.write(f"mtllib {material_path.name}\n")
        obj_file.write("# Vertices\n")
        # original was [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        vertexformat = f"v {{:.{num_decimal_digits}f}} {{:.{num_decimal_digits}f}} {{:.{num_decimal_digits}f}}\n"
        for x, y, z in vertices:
            obj_file.write(vertexformat.format(x, y, z))

        obj_file.write("\n# Faces\n")
        for material, faces_with_material in faces.items():
            obj_file.write(f"usemtl mat{material}\n")
            for a, b, c, d in faces_with_material:
                obj_file.write(f"f {a} {b} {c} {d}\n")



def generateColoredObj(
    reducedModel: np.ndarray, distanceMask: np.ndarray, outputDataPath: Path, bronchusColorMask: np.ndarray | None = None
):
    print(f"Loaded model with shape {reducedModel.shape}")
    print(f"Loaded color mask with shape {distanceMask.shape}")
    if bronchusColorMask is not None:
        bronchus_color_mask = bronchusColorMask["color_mask"]
        bronchus_color_codes = bronchusColorMask["color_codes"]
        print(bronchus_color_codes)
        print(f"Loaded color mask with shape {bronchus_color_mask.shape}")
        color_codes = {i: code for i, code in enumerate(bronchus_color_codes)}
        print(f"Loaded color mask with color codes: {color_codes}")
    else:
        print("WARNING: Color mask not found, using white for all voxels!")
        bronchus_color_mask = None
        color_codes = {0: (1, 1, 1)}

    if not outputDataPath.exists():
        outputDataPath.mkdir(parents=True, exist_ok=True)

    rot_mat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    # Remove lobe coordinates from model by clipping everything
    # between 0 and 2, then modulo everything by 2 to remove 2s
    skeleton = skeletonize(np.clip(reducedModel, 0, 2) % 2)
    generate_obj(outputDataPath / "skeleton.obj", set(), skeleton, rot_mat=rot_mat)

    generate_obj(
        outputDataPath / "bronchus.obj",
        {1},
        reducedModel,
        color_mask=bronchus_color_mask,
        color_to_rgb_tuple=color_codes, # type: ignore
        rot_mat=rot_mat,
    )

    generate_obj(outputDataPath / "distance_mask.obj", {1}, reducedModel, color_mask=distanceMask, rot_mat=rot_mat)
    generate_obj(outputDataPath / "veins.obj", {7}, reducedModel, color_to_rgb_tuple={0: (0, 0, 1)}, rot_mat=rot_mat)
    generate_obj(outputDataPath / "arteries.obj", {8}, reducedModel, color_to_rgb_tuple={0: (1, 0, 0)}, rot_mat=rot_mat)
    # generate_obj(outputDataPath / "lung.obj", set(), model, color_mask=model,
    #              color_to_rgb_tuple=original_color_tuples, rot_mat=rot_mat)


if __name__ == '__main__':
    reducedModelPath = Path(r'E:\conda_projects\Airway\data\customTest\airTree_preprocessed.npz')
    distanceMaskPath = reducedModelPath.parent / "distance_mask.npz"
    bronchusColorMaskPath = reducedModelPath.parent / "bronchus_color_mask.npz"

    generateColoredObj(
        reducedModel=np.load(reducedModelPath)['arr_0'],
        distanceMask=np.load(distanceMaskPath)['arr_0'],
        bronchusColorMask=np.load(bronchusColorMaskPath),
        outputDataPath=reducedModelPath.parent
    )
