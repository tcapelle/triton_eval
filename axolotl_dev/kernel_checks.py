import ast
import weave

# Whitelist of torch attributes/methods allowed in a wrapper function.
# These are primarily for tensor creation, allocation, type/device checks, or shape info.
TORCH_WRAPPER_WHITELIST = {
    # Creation/Allocation
    "empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like",
    "arange", "linspace", "eye", "tensor",
    # Type/Property Checks
    "is_tensor", "is_floating_point", "is_complex", "is_conj", "is_nonzero",
    "dtype", "device", "layout", "ndim", "shape", "size", "stride", "numel", "element_size",
    "get_default_dtype", "set_default_dtype", "get_device", "can_cast",
    # Data type objects/functions often used with .to() or in creation
    "float32", "float", "float64", "double", "float16", "half", "bfloat16",
    "complex32", "complex64", "complex128",
    "int8", "int16", "int32", "int64", "int", "long", "short",
    "uint8", "bool",
    # Casting/Device Transfer
    "to", "cpu", "cuda",
    # Misc
    "as_tensor", "from_numpy", "manual_seed", "is_autocast_enabled", "get_autocast_gpu_dtype"
}

class TritonKernelSanityChecker(ast.NodeVisitor):
    """
    AST visitor to ensure:
    1. No torch.* calls or aliased torch usage inside Triton kernels.
    2. The entrypoint function correctly uses a defined Triton kernel.
    3. The entrypoint wrapper does not perform disallowed torch computations.
    """
    def __init__(self, entrypoint_name: str):
        self.entrypoint_name = entrypoint_name
        self.bad_torch_usage_in_kernel = False  # True if torch.* is called inside a Triton kernel
        self.triton_kernel_definitions = set()  # Names of functions decorated with @triton.jit
        self.entrypoint_is_kernel = False       # True if the entrypoint function itself is a Triton kernel
        self.entrypoint_calls_triton_kernel = False # True if the entrypoint (wrapper) calls a Triton kernel
        self.disallowed_torch_op_in_wrapper = False # True if wrapper uses non-whitelisted torch ops

        # State for AST traversal
        self.current_function_name = None
        self.inside_triton_kernel_definition = False # True when visiting the body of a @triton.jit function

        # For tracking torch aliases
        self.resolved_torch_module_names = set() # e.g., {"torch", "my_torch_module_alias"}
        self.imported_torch_symbols = set()      # e.g., {"add", "empty"} for "from torch import add, empty"
                                                 # and their subsequent aliases like "my_add = add"

    def _scan_for_torch_references(self, tree_body):
        """Scans a list of AST nodes (e.g., module body) for torch imports and assignments to populate alias sets."""
        for item in tree_body:
            if isinstance(item, ast.Import): # import torch or import torch as my_torch
                for alias_node in item.names:
                    if alias_node.name == "torch":
                        self.resolved_torch_module_names.add(alias_node.asname or "torch")
            elif isinstance(item, ast.ImportFrom): # from torch import add, Tensor
                if item.module == "torch":
                    for alias_node in item.names:
                        self.imported_torch_symbols.add(alias_node.asname or alias_node.name)
            elif isinstance(item, ast.Assign): # e.g. my_torch = torch OR my_add = add (if add is from torch)
                if len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    target_name = item.targets[0].id
                    if isinstance(item.value, ast.Name):
                        val_name = item.value.id
                        if val_name in self.resolved_torch_module_names:
                            self.resolved_torch_module_names.add(target_name)
                        elif val_name in self.imported_torch_symbols:
                            # If 'add' is an imported_torch_symbol, and we have 'my_add = add',
                            # then 'my_add' also becomes an imported_torch_symbol for checking purposes.
                            self.imported_torch_symbols.add(target_name)
    
    def visit_Module(self, node):
        self._scan_for_torch_references(node.body) # Scan for global imports/aliases first
        self.generic_visit(node) # Changed from super().visit(node)

    def visit_FunctionDef(self, node):
        original_function_name_state = self.current_function_name
        self.current_function_name = node.name

        # Scan for torch aliases defined *within* this function's scope (less common for kernels)
        # self._scan_for_torch_references(node.body) # Potentially too broad/complex if functions are nested.
                                                    # Sticking to module-level aliases for now.

        is_triton_kernel_decorator = any(
            isinstance(deco, ast.Attribute)
            and isinstance(deco.value, ast.Name)
            and deco.value.id in ("triton", "tl")
            and deco.attr == "jit"
            for deco in node.decorator_list
        )

        if is_triton_kernel_decorator:
            self.triton_kernel_definitions.add(node.name)
            if node.name == self.entrypoint_name:
                self.entrypoint_is_kernel = True

            original_inside_triton_kernel_state = self.inside_triton_kernel_definition
            self.inside_triton_kernel_definition = True
            self.generic_visit(node) # Changed from super().generic_visit(node)
            self.inside_triton_kernel_definition = original_inside_triton_kernel_state
        else:
            # Not a Triton kernel, could be the entrypoint wrapper or other helper functions
            self.generic_visit(node) # Changed from super().generic_visit(node)

        self.current_function_name = original_function_name_state

    def visit_Attribute(self, node): # Handles access like torch.add or my_alias.add
        if self.inside_triton_kernel_definition:
            if isinstance(node.value, ast.Name) and node.value.id in self.resolved_torch_module_names:
                self.bad_torch_usage_in_kernel = True
        
        # Check for disallowed torch attribute access in wrapper (e.g. torch.float32 which is fine vs torch.add which is not)
        # This is more relevant for calls like torch.add(), handled in visit_Call.
        # Attribute access itself (like `torch.float32`) is often for types and usually fine.
        # The TORCH_WRAPPER_WHITELIST is primarily for callable methods/functions.

        self.generic_visit(node) # Changed from super().generic_visit(node)

    def visit_Call(self, node):
        is_problematic_call_in_kernel = False
        is_disallowed_call_in_wrapper = False
        
        # Determine if the call is a torch-related call
        # func_origin_name will be the method name (e.g. 'add' from 'torch.add()')
        # or the imported symbol name (e.g. 'add' from 'from torch import add; add()')
        func_origin_name = None
        is_torch_module_call = False # e.g. torch.add() or alias.add()
        is_imported_torch_symbol_call = False # e.g. add() after from torch import add

        if isinstance(node.func, ast.Attribute): # e.g. torch.add() or my_torch_alias.add()
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.resolved_torch_module_names:
                is_torch_module_call = True
                func_origin_name = node.func.attr # e.g., "add"
        elif isinstance(node.func, ast.Name): # e.g. add() after "from torch import add" or "my_add = add"
            if node.func.id in self.imported_torch_symbols:
                is_imported_torch_symbol_call = True
                func_origin_name = node.func.id # This is the alias name, e.g. "add" or "my_add"
                                                # For whitelist, we rely on aliases not obscuring computation.
                                                # A perfect system would map alias to original torch name.

        # Check 1: Disallowed torch usage inside a Triton kernel
        if self.inside_triton_kernel_definition:
            if is_torch_module_call or is_imported_torch_symbol_call:
                # Any torch module call or direct imported torch symbol call is bad in kernel
                self.bad_torch_usage_in_kernel = True
        
        # Check 2: Disallowed torch computation in the wrapper function
        if self.current_function_name == self.entrypoint_name and \
           not self.entrypoint_is_kernel and \
           func_origin_name is not None: # A torch-related call was identified
            if func_origin_name not in TORCH_WRAPPER_WHITELIST:
                # If it's a known torch call (module or imported symbol) and not in whitelist, flag it.
                self.disallowed_torch_op_in_wrapper = True

        # Check 3: Entrypoint wrapper calling a defined Triton kernel
        if self.current_function_name == self.entrypoint_name and \
           not self.entrypoint_is_kernel and \
           not self.entrypoint_calls_triton_kernel:
            
            called_func_name_str = None
            if isinstance(node.func, ast.Name):
                called_func_name_str = node.func.id
            elif isinstance(node.func, ast.Subscript): # Kernel launch like `kernel_name[grid](...)`
                if isinstance(node.func.value, ast.Name):
                    called_func_name_str = node.func.value.id
            
            if called_func_name_str and called_func_name_str in self.triton_kernel_definitions:
                self.entrypoint_calls_triton_kernel = True
        
        self.generic_visit(node) # Changed from super().generic_visit(node)

@weave.op
def uses_torch_in_kernel(src: str, entrypoint: str) -> bool:
    """
    Parse source, return True (indicating a problem) if:
    1. Any torch.* or aliased torch usage appears inside a @triton.jit kernel's definition.
    2. The `entrypoint` function is not a Triton kernel itself AND no Triton kernels are defined in the source.
    3. The `entrypoint` function is not a Triton kernel itself AND it does not call any defined Triton kernel.
    4. The `entrypoint` function is a wrapper (calls Triton) but also performs non-whitelisted torch computations.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Malformed code, treat as hacked
        return True
    
    checker = TritonKernelSanityChecker(entrypoint_name=entrypoint)
    checker.visit(tree)

    # Rule 1: Bad torch usage directly inside any Triton kernel's definition.
    if checker.bad_torch_usage_in_kernel:
        return True

    # Rule 2: If the entrypoint function IS a Triton kernel itself, it's good (as Rule 1 passed).
    if checker.entrypoint_is_kernel:
        return False 

    # --- At this point, the entrypoint is a wrapper function ---

    # Rule 3a: Wrapper has no Triton kernels defined in the source to call.
    if not checker.triton_kernel_definitions:
        return True
        
    # Rule 3b: Wrapper doesn't call any of the defined Triton kernels.
    if not checker.entrypoint_calls_triton_kernel:
        return True

    # --- At this point, entrypoint is a wrapper AND it calls a defined Triton kernel ---

    # Rule 4: Wrapper calls a Triton kernel, but also performs disallowed torch computations itself.
    if checker.disallowed_torch_op_in_wrapper:
        return True

    return False # All checks passed

class TritonCoverageChecker(ast.NodeVisitor):
    """
    AST visitor to collect which Triton primitives appear in kernels.
    """
    def __init__(self):
        self.primitives = set()
        self.inside_kernel = False

    def visit_FunctionDef(self, node):
        is_kernel = any(
            isinstance(deco, ast.Attribute)
            and isinstance(deco.value, ast.Name)
            and deco.value.id in ("triton", "tl")
            and deco.attr == "jit"
            for deco in node.decorator_list
        )
        if is_kernel:
            self.inside_kernel = True
            self.generic_visit(node) # Changed from super().generic_visit
            self.inside_kernel = False
        else:
            self.generic_visit(node) # Changed from super().generic_visit

    def visit_Attribute(self, node):
        if self.inside_kernel:
            # Attempt to reconstruct full name, e.g., tl.load, tl.constexpr
            qname_parts = []
            curr = node
            while isinstance(curr, ast.Attribute):
                qname_parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                qname_parts.append(curr.id)
                fullname = ".".join(reversed(qname_parts))
                
                # Check for specific Triton primitives we care about
                if fullname.startswith("tl.") and fullname in {
                    "tl.load", "tl.store", "tl.arange", "tl.zeros", "tl.full", 
                    "tl.program", "tl.num_programs", "tl.constexpr", "tl.reduce", 
                    "tl.associative_scan", "tl.dot", "tl.sum", "tl.max", "tl.min",
                    "tl.sqrt", "tl.exp", "tl.log", "tl.sin", "tl.cos" # Add more common ones
                }:
                    self.primitives.add(fullname)

        self.generic_visit(node) # Changed from super().generic_visit

@weave.op
def count_primitives(src: str) -> int:
    """
    Parse source, count distinct Triton primitives used inside @triton.jit kernels.
    """
    try:
        tree = ast.parse(src)
    except Exception: # Catches SyntaxError and other potential parsing issues
        return 0
    checker = TritonCoverageChecker()
    checker.visit(tree)
    return len(checker.primitives) 