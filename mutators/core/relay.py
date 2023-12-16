
import os
import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm.relay.dataflow_pattern import *
import collections
from tvm import te

# Relay IR Transform Pass
@relay.transform.function_pass(opt_level=0)
class MutatorClass:
    def __init__(self, mutation_data):
        self.mutation_data = mutation_data

    def transform_function(self, func, mod, ctx):
        # just for demo purposes
        mutation_data = self.mutation_data
        
        class Mutator(relay.ExprMutator):
            def __init__(self):
                super().__init__()
                self.count = 0

            def prepare_and_call_operation(self, operation, param_input):
                return relay.Call(relay.op.get(operation), [param_input])       

            def prepare_and_call_transpose(self, args, arg_no, shape=None):

                unvisited_arg = args[arg_no]
                visited_arg = self.visit(args[arg_no]) if shape is None else args[arg_no]

                if(type(unvisited_arg) == relay.expr.Call):
                    shape = unvisited_arg.checked_type.shape if shape is None else shape
                    
                elif(hasattr(unvisited_arg, "type_annotation")):
                    shape = unvisited_arg.type_annotation.shape
                else:
                    print("Warning: argument shape could not be determined. Skipping.....")
                    return unvisited_arg

                param_input = visited_arg

                
                
                # Infer tensor dimensions to transpose (same shape dimensions).
                common = [item for item, count in collections.Counter(shape).items() if count > 1]
                pairs = []
                for c in common:
                    pairs.append([i for i, e in enumerate(shape) if e == c])
                    
                if(len(pairs) == 0):
                    #+ str(param_input) + "
                    print("Warning: transpose could not be applied on arg with shape (" + str(shape) + "). Skipping.....")
                    return param_input
                
                transpose_dims = []
                
                
                # min = float('inf')

                # for p in pairs:
                #     if (p[0] < min):
                #         min = p[0]
                #     transpose_dims.append(p[1])
                #     transpose_dims.append(p[0])

                for i in range(0, len(shape)):
                    transpose_dims.insert(i, i)

                for p in pairs:
                    val = transpose_dims[p[0]]
                    transpose_dims[p[0]] = transpose_dims[p[1]]
                    transpose_dims[p[1]] = val

                return relay.transpose(param_input, axes=transpose_dims)
            
            def visit_call(self, call):
                super().visit_call(call)

                mutation_data["metrics"]["total"] += 1
                
                start = mutation_data["start"] if ("start" in mutation_data) else 1
                end  = mutation_data["end"] if ("end" in mutation_data) else -1
                
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                
                mutation_type = mutation_data["type"]

                if(mutation_type == "apply_tensor_operation"):
                    if(not "operation" in mutation_data):
                        raise ValueError("Error: no operation specified")
                    
                    operation = mutation_data["operation"]
                    
                    arg_no = mutation_data["arg"] if "arg" in mutation_data else 0

                    if(call.op.name in mutation_data["functions"]):
                        self.count = self.count + 1
                        if(self.count < start or (end != -1 and self.count > end)):
                            return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                        
                        #print("Count: " + str(self.count))
                        if(arg_no < len(new_args)):
                            
                            if(operation == "transpose"):
                                # Note: It is important NOT to visit args here,
                                # as we want the direct nodes related with this specific function call.
                                if("order" in mutation_data and mutation_data["order"] == "after"):

                                    # Set a new call object with the Op and args visited.
                                    new_call_original = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

                                    # Note: block recursive visiting to avoid stack overflow by explicitly providing shape.
                                    # Since the Call object is explicitly defined and not obtained by the AST, the type checker
                                    # was not able to infer types. Therefore, we indicate to the preparing function to consider the
                                    # new call object for graph construction in relay, but obtain the shape from the original call,
                                    # on which the type checker has run on.
                                    mutation_data["metrics"]["call"] += 1
                                    return self.prepare_and_call_transpose([new_call_original], 0, shape=call.checked_type.shape)
                                else:
                                    call_mutation = self.prepare_and_call_transpose(call.args, arg_no)

                            else:
                                # If non-transpose operation, support call after the original call as well.
                                if("order" in mutation_data and mutation_data["order"] == "after"):
                                    # Set a new call object with the Op and args visited.
                                    new_call_original = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                                    mutation_data["metrics"]["call"] += 1
                                    return self.prepare_and_call_operation(operation, new_call_original)
                                else:
                                    call_mutation = self.prepare_and_call_operation(operation, new_args[arg_no])
                            #Prepare arg with the transformation invocation, and return mutated function call.
                            new_args[arg_no] = call_mutation
                            mutation_data["metrics"]["call"] += 1
                            return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                                
                        else:
                            print("Warning: " + operation + " can not be applied on step " + str(self.count) + " (" + call.op.name + ") in graph. Skipping.....")

                elif(mutation_type == "replace_func"):
                    self.count = self.count + 1
                    if(self.count >= start and (end == -1 or self.count <= end)):
                        original_func_name = mutation_data["original_func_name"]
                        mutated_func_name = mutation_data["mutated_func_name"]
                        if(call.op.name == original_func_name):
                            mutation_data["metrics"]["call"] += 1
                            return relay.Call(relay.op.get(mutated_func_name), new_args)
                
                return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                
        return Mutator().visit(func)


def mutate(mod, mutation_data):
    # Apply Relay IR transformations to model.
    fpass = MutatorClass(mutation_data)
    return fpass(mod)