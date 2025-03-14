import argparse
import json
import os
import sys
import time

import bpy
import numpy as np


def write_done(path: str, mark: str, status: bool):
    file_name = mark
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write(str(status))
                time.sleep(0.01)


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Objaverse glb mesh status calculation start. Local time is %s" %
          (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--done_file_mark', type=str,
                        help='mark of white file')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    mesh_folder = os.path.split(source_mesh_path)[0]
    done_file_mark = args.done_file_mark

    bpy.ops.wm.obj_import(filepath=source_mesh_path)
    meshes = []
    for ind, obj in enumerate(bpy.context.selected_objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    bsdf_material_number = 0
    material_number = 0
    for mesh in meshes:
        if mesh.material_slots:
            for slot in mesh.material_slots:
                material_number = material_number + 1
                material_name = slot.material.name
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        bsdf_material_number = bsdf_material_number + 1

    if bsdf_material_number < int(0.1 * material_number):
        exit(-1)

    white_model = True

    color_value_list = []
    for mesh in meshes:
        if mesh.material_slots:
            for slot in mesh.material_slots:
                material_name = slot.material.name
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Base Color"].links) > 0:
                            white_model = False
                        else:
                            color_value = np.array([node.inputs["Base Color"].default_value[0],
                                                    node.inputs["Base Color"].default_value[1],
                                                    node.inputs["Base Color"].default_value[2],
                                                    node.inputs["Base Color"].default_value[3]])
                            if len(color_value_list) == 0:
                                color_value_list.append(color_value)
                            else:
                                for other_value in color_value_list:
                                    check_same_array = (other_value == color_value)
                                    if check_same_array.all():
                                        break
                                    color_value_list.append(color_value)

    if len(color_value_list) > 3:
        white_model = False

    if white_model:
        print("Model at %s is white......." % (source_mesh_path))
    else:
        print("Model at %s is not white......." % (source_mesh_path))
    write_done(mesh_folder, done_file_mark, white_model)
