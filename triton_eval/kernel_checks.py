import ast
import weave

# Whitelist of torch attributes/methods allowed in the wrapper function.
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
    AST visitor to validate:
    1. No torch calls inside Triton kernels.
    2. Entrypoint wrapper calls a defined Triton kernel and only uses allowed torch ops.
    3. Detect Triton kernels with empty or no-op bodies (i.e., hacking to do nothing).
    """
    def __init__(self, entrypoint_name: str):
        self.entrypoint_name = entrypoint_name
        # Results
        self.bad_torch_usage_in_kernel = False
        self.entrypoint_is_kernel = False
        self.entrypoint_calls_triton_kernel = False
        self.disallowed_torch_op_in_wrapper = False
        self.empty_kernel_detected = False
        # Collected module-level info
        self.torch_aliases = set()
        self.imported_torch_symbols = set()
        self.triton_kernel_defs = set()
        # Traversal state
        self.current_function = None
        self.inside_kernel = False
        self.function_calls = {}  # Maps function name -> set of functions it calls

    def _is_triton_jit(self, deco):
        # Handles @triton.jit and @triton.jit(...)
        if isinstance(deco, ast.Attribute):
            return isinstance(deco.value, ast.Name) and deco.value.id in ("triton", "tl") and deco.attr == "jit"
        if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
            return isinstance(deco.func.value, ast.Name) and deco.func.value.id in ("triton", "tl") and deco.func.attr == "jit"
        return False

    def _collect_module_info(self, node):
        for item in node.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    if alias.name == "torch":
                        self.torch_aliases.add(alias.asname or "torch")
            elif isinstance(item, ast.ImportFrom):
                if item.module == "torch":
                    for alias in item.names:
                        self.imported_torch_symbols.add(alias.asname or alias.name)
            elif isinstance(item, ast.Assign):
                if len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and isinstance(item.value, ast.Name):
                    tgt = item.targets[0].id
                    val = item.value.id
                    if val in self.torch_aliases:
                        self.torch_aliases.add(tgt)
                    elif val in self.imported_torch_symbols:
                        self.imported_torch_symbols.add(tgt)
            elif isinstance(item, ast.FunctionDef):
                for deco in item.decorator_list:
                    if self._is_triton_jit(deco):
                        self.triton_kernel_defs.add(item.name)
                        if item.name == self.entrypoint_name:
                            self.entrypoint_is_kernel = True
                        break

    def _get_call_info(self, node):
        # Returns (name, kind) where kind is 'torch', 'kernel', or None
        func = node.func
        if isinstance(func, ast.Subscript):
            if isinstance(func.value, ast.Name) and func.value.id in self.triton_kernel_defs:
                return func.value.id, 'kernel'
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            val = func.value.id
            attr = func.attr
            if val in self.torch_aliases:
                return attr, 'torch'
            if attr in self.triton_kernel_defs:
                return attr, 'kernel'
        if isinstance(func, ast.Name):
            nm = func.id
            if nm in self.imported_torch_symbols:
                return nm, 'torch'
            if nm in self.triton_kernel_defs:
                return nm, 'kernel'
        return None, None

    def visit_Module(self, node):
        self._collect_module_info(node)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "torch":
                self.torch_aliases.add(alias.asname or "torch")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "torch":
            for alias in node.names:
                self.imported_torch_symbols.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name):
            tgt = node.targets[0].id
            val = node.value.id
            if val in self.torch_aliases:
                self.torch_aliases.add(tgt)
            elif val in self.imported_torch_symbols:
                self.imported_torch_symbols.add(tgt)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        prev_func = self.current_function
        prev_inside = self.inside_kernel
        self.current_function = node.name
        self.function_calls[node.name] = set()  # Initialize empty set of called functions
        
        # Detect and flag empty Triton kernels
        if node.name in self.triton_kernel_defs:
            # consider only non-pass, non-docstring statements as effective
            effective_stmts = [stmt for stmt in node.body
                               if not (isinstance(stmt, ast.Pass) or
                                       (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)))]
            if not effective_stmts:
                self.empty_kernel_detected = True
            self.inside_kernel = True
        self.generic_visit(node)
        self.current_function = prev_func
        self.inside_kernel = prev_inside

    def visit_Call(self, node):
        name, kind = self._get_call_info(node)
        
        # Record this function call in our call graph
        if name and self.current_function:
            self.function_calls.setdefault(self.current_function, set()).add(name)
        # Handle direct function name references that may not be detected by _get_call_info
        elif isinstance(node.func, ast.Name) and self.current_function:
            func_name = node.func.id
            self.function_calls.setdefault(self.current_function, set()).add(func_name)
            
        if self.inside_kernel and kind == 'torch':
            self.bad_torch_usage_in_kernel = True
        if self.current_function == self.entrypoint_name and not self.entrypoint_is_kernel:
            if kind == 'torch' and name not in TORCH_WRAPPER_WHITELIST:
                self.disallowed_torch_op_in_wrapper = True
            if kind == 'kernel':
                self.entrypoint_calls_triton_kernel = True
        self.generic_visit(node)

    def _calls_kernel_indirectly(self, func_name, visited=None):
        """Recursively check if a function calls a Triton kernel through other functions."""
        if visited is None:
            visited = set()
            
        if func_name in visited:
            return False  # Avoid cycles
            
        visited.add(func_name)
        
        # Direct calls to kernels
        calls = self.function_calls.get(func_name, set())
        if any(callee in self.triton_kernel_defs for callee in calls):
            return True
            
        # Indirect calls through other functions
        return any(self._calls_kernel_indirectly(callee, visited) for callee in calls 
                  if callee in self.function_calls)

@weave.op
def is_valid_kernel(src: str, entrypoint: str) -> dict:
    """
    Parse source, return a dictionary indicating if the kernel is valid and why.
    A kernel is considered invalid if:
    1. Any torch.* or aliased torch usage appears inside a @triton.jit kernel's definition.
    2. The `entrypoint` function is not a Triton kernel itself AND no Triton kernels are defined in the source.
    3. The `entrypoint` function is not a Triton kernel itself AND it does not call any defined Triton kernel.
    4. The `entrypoint` function is a wrapper (calls Triton) but also performs non-whitelisted torch computations.
    5. The source code has a syntax error.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        # Malformed code - return generic message without details
        return {'is_valid': False, 'reason': 'Syntax error parsing source code.'}

    checker = TritonKernelSanityChecker(entrypoint_name=entrypoint)
    checker.visit(tree)

    # Rule 1: Bad torch usage directly inside any Triton kernel's definition.
    if checker.bad_torch_usage_in_kernel:
        return {'is_valid': False, 'reason': 'Torch usage detected inside a Triton kernel.'}

    # Rule 2: If the entrypoint function IS a Triton kernel itself, it's good (as Rule 1 passed).
    if checker.entrypoint_is_kernel:
        return {'is_valid': True, 'reason': ''}
    
    if checker.empty_kernel_detected:
        return {'is_valid': False, 'reason': 'Triton kernel has an empty body.'}

    # --- At this point, the entrypoint is a wrapper function ---

    # Rule 3a: Source needs to define Triton kernels to call
    if not checker.triton_kernel_defs:
        return {'is_valid': False, 'reason': 'Entrypoint is a wrapper function, but no Triton kernels are defined in the source.'}

    # Rule 3b: Wrapper must either directly call a kernel or call other functions that eventually call a kernel
    calls_kernel_directly = checker.entrypoint_calls_triton_kernel
    calls_kernel_indirectly = checker._calls_kernel_indirectly(entrypoint)
    
    if not (calls_kernel_directly or calls_kernel_indirectly):
        return {'is_valid': False, 'reason': 'Entrypoint is a wrapper function, but it does not call any defined Triton kernel directly or indirectly.'}

    # --- At this point, entrypoint is a wrapper AND it calls a defined Triton kernel ---

    # Rule 4: Wrapper calls a Triton kernel, but also performs disallowed torch computations itself.
    if checker.disallowed_torch_op_in_wrapper:
        return {'is_valid': False, 'reason': 'Entrypoint is a wrapper function that calls a Triton kernel, but also performs non-whitelisted torch operations.'}

    return {'is_valid': True, 'reason': ''}

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