import bpy
import numpy as np
from . import nodes


def get_transformations_pdbx(file_pdbx):
    import biotite.structure.io.pdbx as pdbx
    # The output transform_dict has an entry for each transformation, indexable by the string
    # integer of the assembly number (e.g. transform_dict.get('1')) which contains tuple of the 3x3 rotation 
    # matrix and the 1x3 transform matrix
    assembly_gen_category = file_pdbx.get_category(
        "pdbx_struct_assembly_gen", expect_looped=True
    )
    struct_oper_category = file_pdbx.get_category(
        "pdbx_struct_oper_list", expect_looped=True
    )
    transformations = pdbx.convert._get_transformations(struct_oper_category)
    # assembly_id = assembly_gen_category["assembly_id"][0]
    # transform_dict = pdbx.convert._get_transformations(file_pdbx.get_category('pdbx_struct_oper_list'))
    # extra_fields_and_asym = ["label_asym_id"]
    # label_asym_id = np.unique(file_pdbx.get_category('atom_site')['label_asym_id'])
    # auth_asym_id = np.unique(file_pdbx.get_category('atom_site')['auth_asym_id'])
    transform_dict = {}
    for aid, op_expr, asym_id_expr in zip(
                    assembly_gen_category["assembly_id"],
                    assembly_gen_category["oper_expression"],
                    assembly_gen_category["asym_id_list"]):
        # skip until biotite fix the issue
        if len(op_expr.split("-")) > 2:
            continue
        operations = pdbx.convert._parse_operation_expression(op_expr)
        asym_ids = asym_id_expr.split(",")
        if aid not in transform_dict:
            transform_dict[aid] = {}
        for asym in asym_ids:
            if asym not in transform_dict[aid]:
                transform_dict[aid][asym] = []
            for i, operation in enumerate(operations):
                for op_step in operation:
                    transform_dict[aid][asym].append(transformations[op_step])
    return transform_dict


def get_transformations_pdb(file_pdb):
    from re import compile
    # get the lines where 'SMTRY' appears, which specify the actual symmetry operations
    sym_lines = np.array(file_pdb.lines)[np.char.rfind(np.array(file_pdb.lines), 'SMTRY') > 0]
    
    regex = compile('\d\.\d+') # find where there is a digit, a decimal, and then multiple more digits
    
    n_mat = int(len(sym_lines) / 4)
    sym_array = np.zeros([len(sym_lines),4], dtype=np.float32)
    
    
    for i in range(len(sym_lines)):
        sym_array[i] = np.array(regex.findall(sym_lines[i]), dtype = np.float32)
    
    transform_dict = []
    
    for i in range(n_mat):
        
        mat_start = i * 3
        mat_end = (i + 1) * 3
        
        transform_dict.append((
            sym_array[mat_start:mat_end, :3], 
            sym_array[mat_start:mat_end, 3:].reshape([1, 3])
        ))
    
    return transform_dict

def get_transformations_mmtf(all_assemblies, world_scale = 0.01):
    
    # get the list of assemblies. Each item in the list is a dictionary which has two components, 
    # the 'transformList' and the 'name'. The name specifies assembly ID (as a string of an integer)
    # the 'transformList' is a dictionary which contains the chain IDs to transform (currently ignored)
    # and the 4x4 transformation matrix which is applied to the asymettric unit to build out the biological 
    # assembly. The output transform_dict has an entry for each transformation, indexable by the string
    # integer of the assembly number (e.g. transform_dict.get('1')) which contains tuple of the 3x3 rotation 
    # matrix and the 1x3 transform matrix

    transform_dict = {}

    for assembly in all_assemblies:
        counter_mat = 0
        for transform in assembly.get('transformList'):
            counter_mat += 1
            # print(transform)
            mat = transform.get('matrix')
            mat = np.array(mat).reshape(4, 4) * world_scale
            
            transform_dict[str(counter_mat)] = (
                mat[:3, :3], 
                mat[:3, 3:].reshape(1, 3)[0]
            )
    return transform_dict

def create_assembly_node(name, transform_dict_string):
    
    # from json import loads
    
    # transform_dict = loads(transform_dict_string)
    transform_dict = transform_dict_string
    
    node_mat = bpy.data.node_groups.get('MOL_RotTransMat_' + name)
    if node_mat:
        return node_mat
    
    node_mat = nodes.gn_new_group_empty('MOL_RotTransMat_' + name)
    node_mat.inputs.remove(node_mat.inputs['Geometry'])
    node_mat.nodes['Group Output'].location = [800, 0]
    node_mat.outputs['Geometry'].name = 'RotTransMat'
    
    counter = 0
    node_transform_list = []
    for transform in transform_dict.values():
        counter =+ 1
        node = nodes.rotation_matrix(
            node_group = node_mat, 
            mat_rot= transform[0], 
            mat_trans= transform[1], 
            location= [0, 0 - (300 * counter)]
        )
        
        node_transform_list.append(node)
    
    node_transform_list.reverse()
    
    node_join = node_mat.nodes.new('GeometryNodeJoinGeometry')
    node_join.location = [300, 0]
    
    for node_transform in node_transform_list:
        node_mat.links.new(node_transform.outputs['Geometry'], node_join.inputs['Geometry'])
    
    node_mat.links.new(node_join.outputs['Geometry'], node_mat.nodes['Group Output'].inputs['RotTransMat'])
    
    return node_mat

def create_biological_assembly_node(name, transform_dict):
    
    node_bio = bpy.data.node_groups.get('MOL_assembly_' + name)
    if node_bio:
        return node_bio
    
    # try to create the assembly transformation nodes first, so 
    # if they fail, nothing else is created
    data_trans = create_assembly_node(name, transform_dict)
    
    node_bio = nodes.gn_new_group_empty('MOL_assembly_' + name)
    
    node_input = node_bio.nodes['Group Input']
    node_output = node_bio.nodes['Group Output']
    
    
    node_output.location = [400, 0]
    node_output.inputs['Geometry'].name = 'Instances'
    
    node_assembly = nodes.add_custom_node_group_to_node(node_bio, 'MOL_utils_bio_assembly', location=[0, 0])
    
    node_trans = nodes.add_custom_node_group_to_node(node_bio, data_trans.name, location = [-400, -200])
    
    link = node_bio.links.new
    
    link(node_input.outputs['Geometry'], node_assembly.inputs['Geometry'])
    link(node_trans.outputs['RotTransMat'], node_assembly.inputs['RotTransMat'])
    link(node_assembly.outputs['Instances'], node_output.inputs['Instances'])
    
    inputs = (
        {'name': 'Scale Rotation', 
         'type': 'NodeSocketFloat', 
         'default': 1},
        {'name': 'Scale Translation', 
         'type': 'NodeSocketFloat', 
         'default': 1}
    )
    
    for input in inputs:
        name = input.get('name')
        type = input.get('type')
        default = input.get('default')
        
        node_bio.inputs.new(type, name)
        node_bio.inputs.get(name).default_value = default
        
        link(node_input.outputs[name], node_assembly.inputs[name])
    
    return node_bio
