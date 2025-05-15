import ast
import weave

class TritonKernelSanityChecker(ast.NodeVisitor):
    """
    AST visitor to ensure no torch.* calls inside Triton kernels.
    """
    def __init__(self):
        self.inside_kernel = False
        self.bad_usage = False
        self.call_called = False

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
            self.generic_visit(node)
            self.inside_kernel = False
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node):
        if self.inside_kernel and isinstance(node.value, ast.Name) and node.value.id == "torch":
            self.bad_usage = True
        self.generic_visit(node)

    def visit_Call(self, node):
        if self.inside_kernel and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                self.bad_usage = True
        elif not self.inside_kernel and isinstance(node.func, ast.Call) and node.func.id == "call":
            self.call_called = True
        self.generic_visit(node)

@weave.op
def uses_torch_in_kernel(src: str) -> bool:
    """
    Parse source, return True if any torch.* usage appears inside a @triton.jit kernel.
    """
    try:
        tree = ast.parse(src)
    except Exception:
        # malformed code, treat as hacked
        return True
    checker = TritonKernelSanityChecker()
    checker.visit(tree)
    return checker.bad_usage or not checker.call_called

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
            self.generic_visit(node)
            self.inside_kernel = False

    def visit_Attribute(self, node):
        if self.inside_kernel:
            fullname = f"{getattr(node.value, 'id', '')}.{node.attr}"
            if fullname in {"tl.load", "tl.store", "tl.arange", "tl.zeros", "tl.full"}:
                self.primitives.add(fullname)
            if isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr in {"program_id", "block_id", "block_size"}:
                self.primitives.add(f"triton.{node.attr}")
        self.generic_visit(node)

@weave.op
def count_primitives(src: str) -> int:
    """
    Parse source, count distinct Triton primitives used inside @triton.jit kernels.
    """
    try:
        tree = ast.parse(src)
    except Exception:
        # malformed code, no primitives found
        return 0
    checker = TritonCoverageChecker()
    checker.visit(tree)
    return len(checker.primitives) 