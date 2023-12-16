
import os
import onnx
import numpy as np
import tvm
import json
from tvm import tir
from tvm import te

# Prepare TIR mutations.


expr_prefix = "tvm.tir.expr"
stmt_prefix = "tvm.tir.stmt"

script_dir = os.path.dirname(os.path.realpath(__file__))

def get_code_dir_path(model_name, out_path_relative):
    return script_dir + "/../../" + out_path_relative + "/" + model_name + "_code"

# Function to collect TIR statements.

elems = {}
elems["if_then_else_stmts"] = []
elems["attr_stmts"] = []
elems["stores"] = []
elems["buffer_stores"] = []
elems["producer_stores"] = []
elems["buffers"] = []
elems["while_loops"] = []
elems["for_loops"] = []
elems["total_ops"] = 0

def find_elements(op):
    # print(dir(tir))
    elems["total_ops"] += 1
    if isinstance(op, tir.IfThenElse):
        elems["if_then_else_stmts"].append(op)
    elif isinstance(op, tir.AttrStmt):
        elems["attr_stmts"].append(op)
    elif isinstance(op, tir.Store):
        elems["stores"].append(op)
    elif isinstance(op, tir.BufferStore):
        elems["buffer_stores"].append(op)
    elif isinstance(op, tir.ProducerStore):
        elems["producer_stores"].append(op)
    elif isinstance(op, tir.Buffer):
        elems["buffers"].append(op)
    elif isinstance(op, tir.For):
        elems["for_loops"].append(op)
    # elif isinstance(op, tir.While):
    #     elems["while_loops"].append(op)


def find_in_attrs(var_name):
    for op in attr_stmts:
        if(hasattr(op.node, "var") and op.node.var.name == var_name):
            return op
    return None

def get_store(store_type):
    return {
        "tir.Store": elems["stores"],
        "tir.BufferStore": elems["buffer_stores"],
        "tir.ProducerStore": elems["producer_stores"]
    }[store_type]



def calculate_value(lvalue, op, rvalue):

    if (op == "/" and rvalue == 0):
        raise ValueError("Attempted division with zero.")

    dict = {
        "+": lvalue + rvalue,
        "-": lvalue - rvalue,
        "*": lvalue * rvalue,
        "/": lvalue / rvalue
    }

    if (op not in dict):
        return lvalue
    else:
        return dict[op]


def filter_mutations(op, mutation_data):
    start = mutation_data["start"] if ("start" in mutation_data) else 1
    end  = mutation_data["end"] if ("end" in mutation_data) else -1
    mutation_data["metrics"]["op"] += 1
    
    if (mutation_data["metrics"]["op"] < start or (end != -1 and mutation_data["metrics"]["op"] > end)):
        return None
    
    # print(mutation_data["metrics"]["op"])
    mutation_data["metrics"]["ops_affected"] += 1
    return op

def prepare_mutations(mutation_data):

    # Function to apply mutations.
    def apply_mutations(op):
        mutation_data["metrics"]["total"] += 1
        #print(filter_mutations)
        
        #print(mutation_data["metrics"]["total"])

        #start = mutation_data["start"] if ("start" in mutation_data) else 1
        #end  = mutation_data["end"] if ("end" in mutation_data) else -1

        #mutate.mutation_data["count"] = mutate.mutation_data["count"] + 1
        #print(mutation_data["metrics"]["op"])

        
        #print ("Mutation data count: " + str(mutation_data["metrics"]["op"]))
        #print(dir(op))
        # TODO: Set order to count in order to be able to select occurences.
        if(mutation_data["type"] == "change_ifstmt"):
            
            if op in elems["if_then_else_stmts"]:

                if (isinstance(op.condition, eval(mutation_data["original"]))):
                    mutation_type = eval(mutation_data["mutation"])
                    op.condition = mutation_type(op.condition.a, op.condition.b)
                    # mutation_data["metrics"]["op"] += 1
                    return filter_mutations(tvm.tir.IfThenElse(op.condition, op.then_case, op.else_case, op.span), mutation_data)

        # While Loops - althout documented, they are not added to tir.            
        # elif(mutation_data["type"] == "change_while"):
        #     if op in elems["while_loops"]:

        #         if (isinstance(op.condition, eval(mutation_data["original"]))):
        #             mutation_type = eval(mutation_data["mutation"])
        #             op.condition = mutation_type(op.condition.a, op.condition.b)
        #             # mutation_data["metrics"]["op"] += 1
        #             return filter_mutations(tir.While(op.condition, op.body, op.span), mutation_data)


        elif (mutation_data["type"] == "change_threshold"):
            if op in elems["if_then_else_stmts"]:
                op_data = mutation_data["op"]

                if (isinstance(op.condition, eval(op_data["condition"]))):
                    threshold = tvm.tir.FloatImm(dtype="float32", value=op_data["threshold_change"])
                    
                    float_a = tvm.tir.Cast(dtype="float32", value=op.condition.a)
                    float_b = tvm.tir.Cast(dtype="float32", value=op.condition.b)

                    condition = eval(op_data["condition"])
                    if(op_data["part"] == "left"):
                        if (op_data["action"] == "add"):
                            new_a = tvm.tir.Add(float_a, threshold)
                        elif (op_data["action"] == "add_perc"):
                            new_a = tvm.tir.Add(float_a, tvm.tir.Mul(float_a, threshold))
                        elif (op_data["action"] == "sub"):
                            new_a = tvm.tir.Sub(float_a, threshold)
                        elif (op_data["action"] == "sub_perc"):
                            new_a = tvm.tir.Add(float_a, tvm.tir.Mul(float_a, threshold))
                        elif (op_data["action"] == "mul"):
                            new_a = tvm.tir.Mul(float_a, threshold)
                        # TODO: Add handling for zero division.
                        elif (op_data["action"] == "div"):
                            new_a = tvm.tir.Div(float_a, threshold)
                        # condition = eval(op_data["condition"])
                        op.condition = condition(new_a, float_b)
                        # mutation_data["metrics"]["op"] += 1
                        return filter_mutations(tvm.tir.IfThenElse(op.condition, op.then_case, op.else_case, op.span), mutation_data)

                    elif(op_data["part"] == "right"):
                        if (op_data["action"] == "add"):
                            new_b = tvm.tir.Add(float_b, threshold)
                        elif (op_data["action"] == "add_perc"):
                            new_b = tvm.tir.Add(float_b, tvm.tir.Mul(float_b, threshold))
                        elif (op_data["action"] == "sub"):
                            new_b = tvm.tir.Sub(float_b, threshold)
                        elif (op_data["action"] == "sub_perc"):
                            new_b = tvm.tir.Sub(float_b, tvm.tir.Mul(float_b, threshold))
                        elif (op_data["action"] == "mul"):
                            new_b = tvm.tir.Mul(float_b, threshold)
                        # TODO: Add handling for zero division.
                        elif (op_data["action"] == "div"):
                            new_b = tvm.tir.Div(float_b, threshold)

                        op.condition = condition(float_a, new_b)
                        # mutation_data["metrics"]["op"] += 1
                        return filter_mutations(tvm.tir.IfThenElse(op.condition, op.then_case, op.else_case, op.span), mutation_data)
                    else:
                        return None

        elif(mutation_data["type"] == "change_type"):
            if op in get_store(mutation_data["stmt_type"]):                          
                if(op.value.dtype == mutation_data["original"]):
                    store_type = eval(mutation_data["stmt_type"])
                    # mutation_data["metrics"]["op"] += 1
                    return filter_mutations(store_type(op.buffer_var, op.value.astype(mutation_data["mutation"]), op.index, op.predicate, op.span), mutation_data)
                return op
        elif(mutation_data["type"] == "change_store_value"):
            if op in get_store(mutation_data["stmt_type"]):                          
                new_value = calculate_value(op.value, mutation_data["operator"], float(mutation_data["modifier_value"]))
                store_type = eval(mutation_data["stmt_type"])
                # mutation_data["metrics"]["op"] += 1
                return filter_mutations(store_type(op.buffer_var, op.value, op.index, op.predicate, op.span), mutation_data)

        elif(mutation_data["type"] == "change_block"):

            if op in elems["attr_stmts"]:

                # TODO: Utilize find_in_attrs(var_name) in order to swap dimensions across variables.
                # if(mutation_data["mutation"] == "swap_dimensions"):
                #     mutation_data["first_var"]


                # if((hasattr(op.node, "var") and (op.node.var.name == mutation_data["var_name"] or mutation_data["var_name"] == "*")) and op.attr_key == mutation_data["attr_key"]):
                #     newValue = tvm.tir.IntImm(dtype="int32", value=int(op.value // 2 if op.value > 1 else 1))
                #     return tvm.tir.AttrStmt(op.node, op.attr_key, newValue, op.body, op.span)
                #     pass
                # Locate the proper variable (or wildcard) - and then (partial evaluation) the proper attribute value.
                if((hasattr(op.node, "var") and (op.node.var.name == mutation_data["var_name"] or mutation_data["var_name"] == "*")) and op.attr_key == mutation_data["attr_key"]):
                    
                    # if (mutation_data["action"] == "add"):
                    #     newValue = (tvm.tir.Add(op.value, mutation_data["attr_value"]))
                    # elif (mutation_data["action"] == "subtract"):
                    #     newValue = (tvm.tir.Sub(op.value, mutation_data["attr_value"]))
                    # else:
                    try:
                        newValue = tvm.tir.IntImm(dtype="int32", value=int(mutation_data["attr_value"]))
                        mutation_data["metrics"]["op"] += 1
                        return filter_mutations(tvm.tir.AttrStmt(op.node, op.attr_key, newValue, op.body, op.span), mutation_data)
                    except:
                        print("Issue with value in block change. Mutation skipped.")
                        
        elif(mutation_data["type"] == "change_loop_threshold"):
            if op in elems["for_loops"]:
                boundary = mutation_data["boundary"]
                action = mutation_data["action"]
                isForKind = "forKind" in mutation_data and mutation_data["forKind"]
                threshold = tvm.tir.FloatImm(dtype="float32", value=mutation_data["threshold_change"])
                # print(dir(op))
                min_val = tvm.tir.Cast(dtype="float32", value=op.min)
                extent = tvm.tir.Cast(dtype="float32", value=op.extent)
                if(boundary == "min_val"):
                    if(action == "add"):
                        min_val = tvm.tir.Add(min_val, threshold)
                    elif(action == "subtract"):
                        min_val = tvm.tir.Sub(min_val, threshold)
                    op.min = min_val
                elif(boundary == "extent"):
                    if(action == "add"):
                        extent = tvm.tir.Add(extent, threshold)
                    elif(action == "subtract"):
                        extent = tvm.tir.Sub(extent, threshold)
                    op.extent = extent
                if(not isForKind):
                    return filter_mutations(tvm.tir.For(op.loop_var, op.min, op.extent, op.kind, op.body, op.thread_binding, op.annotations, op.span), mutation_data)
                else:
                    filter_mutations(tvm.tir.ForKind(op.loop_var, op.min, op.extent, op.kind, op.body, op.thread_binding, op.annotations, op.span), mutation_data)

        return None
        
    return apply_mutations


# TIR Transform Pass.
@tvm.tir.transform.prim_func_pass(opt_level=0)
def mutate(f, mod, ctx):

    global elems

    model_dir_path = get_code_dir_path(mutate.mutation_data["model_name"], mutate.mutation_data["out_path_relative"])
    os.makedirs(model_dir_path, exist_ok=True)
    output_tir_file = open(model_dir_path + "/output_tir.txt",'a+')

    
    mutation_data = {}
    
    # Prepare transformations.
    
    tvm.tir.stmt_functor.post_order_visit(f.body, find_elements)

    # TODO: Refactor
    #mutate.mutation_data["metrics"]["total"] = elems["total_ops"]
    #len(elems["if_then_else_stmts"]) + len(elems["attr_stmts"]) + len(elems["stores"]) + len(elems["buffer_stores"]) + len(elems["producer_stores"]) + len(elems["buffers"]) + len(elems["loops"])

    # Apply transformations
    #mutate.mutation_data["metrics"]["total"] = 0
    newF = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, prepare_mutations(mutate.mutation_data), [mutate.mutation_data["stmt_type"]]))
    
    print(newF, file=output_tir_file)
    # Save TIR to file.
    return newF


@tvm.tir.transform.prim_func_pass(opt_level=0)
def save_original_pass(f, mod, ctx):

    tvm.tir.stmt_functor.post_order_visit(f.body, find_elements)
    
    model_dir_path = get_code_dir_path(save_original_pass.mutation_data["model_name"], save_original_pass.mutation_data["out_path_relative"])
    os.makedirs(model_dir_path, exist_ok=True)
    output_tir_file = open(model_dir_path + "/output_tir.txt",'a+')

    print(f, file=output_tir_file)
    # Save TIR to file.
    return f