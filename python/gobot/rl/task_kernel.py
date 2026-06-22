"""Warp-style Python task kernels for Gobot RL JIT compilation."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
import inspect
import textwrap
from typing import Any, Callable

import numpy as np


# Public kernel surface -----------------------------------------------------
#
# This module is the Python frontend for Gobot task kernels.  It intentionally
# does not compile anything by itself; TaskJitCompiler owns the LLVM step.  The
# job here is closer to Warp's frontend: capture a restricted Python function,
# validate the AST, and lower it to a tiny C++ kernel over flat batch buffers.


class TaskKernelCompileError(RuntimeError):
    """Raised when a Python task kernel uses unsupported syntax."""


@dataclass(frozen=True)
class TaskKernel:
    """Restricted Python function that can be lowered to a Gobot C++ task kernel."""

    func: Callable[..., Any]
    source: str
    lineno: int
    tree: ast.Module
    launch: str = "actor_obs"
    launch_axis: int = 0

    @property
    def name(self) -> str:
        return self.func.__name__

    @property
    def cache_key_source(self) -> str:
        return f"launch={self.launch}:{self.launch_axis}\n{self.source}"


def task_kernel(
    func: Callable[..., Any] | None = None,
    *,
    launch: str = "actor_obs",
    launch_axis: int = 0,
) -> TaskKernel | Callable[[Callable[..., Any]], TaskKernel]:
    """Decorate a restricted Python task function for JIT compilation.

    The lowered C++ kernel executes once per row of ``launch`` along
    ``launch_axis``. Inside the function, ``tid()`` returns that row index.
    """

    def wrap(inner: Callable[..., Any]) -> TaskKernel:
        source, lineno, tree = _extract_function_source(inner)
        return TaskKernel(
            func=inner,
            source=source,
            lineno=lineno,
            tree=tree,
            launch=str(launch),
            launch_axis=int(launch_axis),
        )

    if func is None:
        return wrap
    return wrap(func)


kernel = task_kernel


def tid() -> int:
    raise RuntimeError("gobot.rl.tid() is only valid inside a @task_kernel function")


def dim(array: Any, axis: int) -> int:
    raise RuntimeError("gobot.rl.dim() is only valid inside a @task_kernel function")


def where(condition: bool, true_value: Any, false_value: Any) -> Any:
    raise RuntimeError("gobot.rl.where() is only valid inside a @task_kernel function")


def param(values: Any, index: int, fallback: float) -> float:
    raise RuntimeError("gobot.rl.param() is only valid inside a @task_kernel function")


def flag(values: Any, index: int) -> float:
    raise RuntimeError("gobot.rl.flag() is only valid inside a @task_kernel function")


def weight(values: Any, index: int) -> float:
    raise RuntimeError("gobot.rl.weight() is only valid inside a @task_kernel function")


def generate_cpp_from_task_kernel(
    kernel: TaskKernel,
    arrays: Any,
    *,
    abi_version: int,
) -> tuple[str, tuple[str, ...]]:
    lowerer = _TaskKernelLowerer(kernel, arrays, abi_version=abi_version)
    return lowerer.compile()


def _extract_function_source(func: Callable[..., Any]) -> tuple[str, int, ast.Module]:
    try:
        source_lines, lineno = inspect.getsourcelines(func)
    except OSError as error:
        raise TaskKernelCompileError("task kernels must be defined in a real Python source file") from error
    source = textwrap.dedent("".join(source_lines))
    return source, lineno, ast.parse(source)


class _TaskKernelLowerer:
    """Lower one decorated Python function to Gobot's task-kernel C ABI."""

    def __init__(self, kernel: TaskKernel, arrays: Any, *, abi_version: int) -> None:
        self.kernel = kernel
        self.arrays = arrays
        self.abi_version = int(abi_version)
        self.locals: set[str] = set()
        self.array_names: OrderedDict[str, None] = OrderedDict()
        self.arg_name = ""
        self.constants = _constant_namespace(kernel.func)

    def compile(self) -> tuple[str, tuple[str, ...]]:
        function = self._function_def()
        if not function.args.args:
            raise TaskKernelCompileError(f"task kernel {function.name!r} must accept an array namespace argument")
        self.arg_name = function.args.args[0].arg
        self.array_names.setdefault(self.kernel.launch, None)
        body_lines = self._block(function.body, indent=2)
        pointer_lines = self._pointer_lines()
        source = "\n".join(
            (
                'extern "C" {',
                "",
                _CPP_ABI,
                "",
                _CPP_HELPERS,
                "",
                "int gobot_task_kernel_v1(TaskKernelContext* context) {",
                f"    if (context == nullptr || context->abi_version != {self.abi_version}) {{ return -1; }}",
                "    TaskKernelContext& ctx = *context;",
                f'    const size_t _launch_dim = dim(ctx, "{self.kernel.launch}", {int(self.kernel.launch_axis)});',
                *pointer_lines,
                "    for (size_t _env_id = 0; _env_id < _launch_dim; ++_env_id) {",
                *body_lines,
                "    }",
                "    return 0;",
                "}",
                "",
                "}",
                "",
            )
        )
        return source, tuple(self.array_names.keys())

    def _function_def(self) -> ast.FunctionDef:
        for node in self.kernel.tree.body:
            if isinstance(node, ast.FunctionDef):
                return node
        raise TaskKernelCompileError("task kernel source does not contain a function definition")

    def _block(self, statements: list[ast.stmt], *, indent: int) -> tuple[str, ...]:
        lines: list[str] = []
        for statement in statements:
            lines.extend(self._statement(statement, indent=indent))
        return tuple(lines)

    def _statement(self, statement: ast.stmt, *, indent: int) -> tuple[str, ...]:
        prefix = "    " * indent
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            return ()
        if isinstance(statement, ast.Pass):
            return ()
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                raise TaskKernelCompileError("task kernels only support single-target assignment")
            target = statement.targets[0]
            value = self._expr(statement.value)
            if isinstance(target, ast.Name):
                if target.id in self.locals:
                    return (f"{prefix}{target.id} = {value};",)
                self.locals.add(target.id)
                return (f"{prefix}auto {target.id} = {value};",)
            return (f"{prefix}{self._target(target)} = {value};",)
        if isinstance(statement, ast.AugAssign):
            return (f"{prefix}{self._target(statement.target)} {self._op(statement.op)}= {self._expr(statement.value)};",)
        if isinstance(statement, ast.For):
            if not isinstance(statement.target, ast.Name):
                raise TaskKernelCompileError("task kernels only support simple loop variables")
            start, stop = self._range(statement.iter)
            loop_var = statement.target.id
            old_locals = set(self.locals)
            self.locals.add(loop_var)
            body = self._block(statement.body, indent=indent + 1)
            lines = [f"{prefix}for (size_t {loop_var} = {start}; {loop_var} < {stop}; ++{loop_var}) {{"]
            lines.extend(body)
            lines.append(f"{prefix}}}")
            self.locals = old_locals
            return tuple(lines)
        if isinstance(statement, ast.If):
            old_locals = set(self.locals)
            lines = [f"{prefix}if ({self._expr(statement.test)}) {{"]
            lines.extend(self._block(statement.body, indent=indent + 1))
            self.locals = set(old_locals)
            if statement.orelse:
                lines.append(f"{prefix}}} else {{")
                lines.extend(self._block(statement.orelse, indent=indent + 1))
                self.locals = set(old_locals)
            lines.append(f"{prefix}}}")
            self.locals = old_locals
            return tuple(lines)
        if isinstance(statement, ast.Return):
            if statement.value is None:
                return ()
            raise TaskKernelCompileError("task kernels must not return values")
        raise TaskKernelCompileError(f"unsupported task kernel statement: {type(statement).__name__}")

    def _expr(self, expression: ast.expr) -> str:
        if isinstance(expression, ast.Constant):
            return self._constant(expression.value)
        if isinstance(expression, ast.Name):
            if expression.id in self.locals:
                return expression.id
            if expression.id in self.constants:
                return self._constant(self.constants[expression.id])
            if expression.id in _BUILTIN_FUNCTION_NAMES:
                return expression.id
            raise TaskKernelCompileError(
                f"unknown scalar name {expression.id!r}; use a local variable or a module-level int/float/bool constant"
            )
        if isinstance(expression, ast.Attribute):
            return self._array_attribute_name(expression)
        if isinstance(expression, ast.BinOp):
            return f"({self._expr(expression.left)} {self._op(expression.op)} {self._expr(expression.right)})"
        if isinstance(expression, ast.UnaryOp):
            if isinstance(expression.op, ast.USub):
                return f"(-{self._expr(expression.operand)})"
            if isinstance(expression.op, ast.Not):
                return f"(!{self._expr(expression.operand)})"
        if isinstance(expression, ast.BoolOp):
            op = " && " if isinstance(expression.op, ast.And) else " || "
            return "(" + op.join(self._expr(value) for value in expression.values) + ")"
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                raise TaskKernelCompileError("task kernels only support single comparisons")
            return f"({self._expr(expression.left)} {self._cmp(expression.ops[0])} {self._expr(expression.comparators[0])})"
        if isinstance(expression, ast.IfExp):
            return f"({self._expr(expression.test)} ? {self._expr(expression.body)} : {self._expr(expression.orelse)})"
        if isinstance(expression, ast.Call):
            return self._call(expression)
        if isinstance(expression, ast.Subscript):
            return self._target(expression)
        raise TaskKernelCompileError(f"unsupported task kernel expression: {type(expression).__name__}")

    def _call(self, call: ast.Call) -> str:
        if isinstance(call.func, ast.Name) and call.func.id == "tid":
            if call.args:
                raise TaskKernelCompileError("tid() does not accept arguments")
            return "_env_id"
        if not isinstance(call.func, ast.Name):
            raise TaskKernelCompileError("task kernels only support direct function calls")
        name = call.func.id
        if name == "dim":
            if len(call.args) != 2:
                raise TaskKernelCompileError("dim() expects an array and an axis")
            array_name = self._array_name_expr(call.args[0])
            return f'dim(ctx, "{array_name}", {self._expr(call.args[1])})'
        args = [self._expr(arg) for arg in call.args]
        if name == "where":
            if len(args) != 3:
                raise TaskKernelCompileError("where() expects 3 arguments")
            return f"({args[0]} ? {args[1]} : {args[2]})"
        if name == "sqrt":
            return f"__builtin_sqrtf({', '.join(args)})"
        if name == "abs":
            return f"__builtin_fabsf({', '.join(args)})"
        if name == "exp":
            return f"__builtin_expf({', '.join(args)})"
        if name == "log1p":
            return f"__builtin_log1pf({', '.join(args)})"
        if name == "sin":
            return f"__builtin_sinf({', '.join(args)})"
        if name == "cos":
            return f"__builtin_cosf({', '.join(args)})"
        if name == "copysign":
            return f"__builtin_copysignf({', '.join(args)})"
        if name == "max":
            return f"gobot_max({', '.join(args)})"
        if name == "min":
            return f"gobot_min({', '.join(args)})"
        if name in {"param", "flag", "weight"}:
            return f"{name}({', '.join(args)})"
        if name == "float":
            if len(args) != 1:
                raise TaskKernelCompileError("float() expects one argument")
            return args[0]
        raise TaskKernelCompileError(f"unsupported task kernel call: {name}")

    def _target(self, target: ast.expr) -> str:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Attribute):
                name = self._array_attribute_name(target.value)
                return f"{name}[{self._expr(target.slice)}]"
            if isinstance(target.value, ast.Name):
                return f"{target.value.id}[{self._expr(target.slice)}]"
        raise TaskKernelCompileError(f"unsupported assignment target: {type(target).__name__}")

    def _array_name_expr(self, expression: ast.expr) -> str:
        if isinstance(expression, ast.Attribute):
            return self._array_attribute_name(expression)
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            self.array_names.setdefault(expression.value, None)
            return expression.value
        raise TaskKernelCompileError("array arguments must be written as the kernel namespace attribute, e.g. a.actor_obs")

    def _array_attribute_name(self, expression: ast.Attribute) -> str:
        if isinstance(expression.value, ast.Name) and expression.value.id == self.arg_name:
            self.array_names.setdefault(expression.attr, None)
            return expression.attr
        raise TaskKernelCompileError("task kernels only support array attributes on the first argument")

    def _range(self, expression: ast.expr) -> tuple[str, str]:
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != "range":
            raise TaskKernelCompileError("task kernels only support for loops over range()")
        if len(expression.args) == 1:
            return "0", self._expr(expression.args[0])
        if len(expression.args) == 2:
            return self._expr(expression.args[0]), self._expr(expression.args[1])
        raise TaskKernelCompileError("range() in task kernels supports one or two arguments")

    def _pointer_lines(self) -> tuple[str, ...]:
        lines = []
        missing = []
        for name in self.array_names:
            if not hasattr(self.arrays, name):
                raise TaskKernelCompileError(f"native arrays are missing task kernel buffer {name!r}")
            array = np.asarray(getattr(self.arrays, name))
            accessor = _dtype_accessor(array, name)
            lines.append(f'    auto* {name} = {accessor}(ctx, "{name}");')
            missing.append(f'!has_array_data(ctx, "{name}", {name})')
        if missing:
            lines.append(f"    if ({' || '.join(missing)}) {{ return -3; }}")
        return tuple(lines)

    def _constant(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value!r}f"
        if isinstance(value, int):
            return str(value)
        raise TaskKernelCompileError(f"unsupported task kernel constant {value!r}")

    def _op(self, op: ast.operator) -> str:
        if isinstance(op, ast.Add):
            return "+"
        if isinstance(op, ast.Sub):
            return "-"
        if isinstance(op, ast.Mult):
            return "*"
        if isinstance(op, ast.Div):
            return "/"
        if isinstance(op, ast.Mod):
            return "%"
        raise TaskKernelCompileError(f"unsupported operator: {type(op).__name__}")

    def _cmp(self, op: ast.cmpop) -> str:
        if isinstance(op, ast.Lt):
            return "<"
        if isinstance(op, ast.LtE):
            return "<="
        if isinstance(op, ast.Gt):
            return ">"
        if isinstance(op, ast.GtE):
            return ">="
        if isinstance(op, ast.Eq):
            return "=="
        if isinstance(op, ast.NotEq):
            return "!="
        raise TaskKernelCompileError(f"unsupported comparison: {type(op).__name__}")


def _constant_namespace(func: Callable[..., Any]) -> dict[str, int | float | bool]:
    values: dict[str, int | float | bool] = {}
    try:
        closure_vars = inspect.getclosurevars(func)
    except TypeError:
        closure_vars = None
    namespaces: list[dict[str, Any]] = []
    if closure_vars is not None:
        namespaces.extend([dict(closure_vars.globals), dict(closure_vars.nonlocals)])
    else:
        namespaces.append(dict(getattr(func, "__globals__", {})))
    for namespace in namespaces:
        for name, value in namespace.items():
            if isinstance(value, bool) or isinstance(value, (int, float)):
                values[name] = value
    return values


def _dtype_accessor(array: np.ndarray, name: str) -> str:
    dtype = np.dtype(array.dtype)
    if dtype == np.dtype(np.float32):
        return "f32"
    if dtype == np.dtype(np.uint8):
        return "u8"
    raise TaskKernelCompileError(f"task kernel only supports float32/uint8 buffers, got {dtype} for {name!r}")


_BUILTIN_FUNCTION_NAMES = {
    "abs",
    "copysign",
    "cos",
    "dim",
    "exp",
    "flag",
    "float",
    "log1p",
    "max",
    "min",
    "param",
    "sin",
    "sqrt",
    "tid",
    "weight",
    "where",
}


_CPP_ABI = r"""
using size_t = decltype(sizeof(0));
using uint32_t = unsigned int;
using uint8_t = unsigned char;

struct TaskArrayView {
    const char* name;
    uint32_t dtype;
    uint32_t rank;
    size_t shape[4];
    size_t strides[4];
    void* data;
};

struct TaskKernelContext {
    uint32_t abi_version;
    size_t array_count;
    const TaskArrayView* arrays;
};
"""


_CPP_HELPERS = r"""
static inline bool streq(const char* a, const char* b) {
    while (*a != 0 && *b != 0) {
        if (*a != *b) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

static inline float gobot_max(float a, float b) {
    return a > b ? a : b;
}

static inline float gobot_min(float a, float b) {
    return a < b ? a : b;
}

static const TaskArrayView* find_array(const TaskKernelContext& ctx, const char* name) {
    for (size_t i = 0; i < ctx.array_count; ++i) {
        if (ctx.arrays[i].name != nullptr && streq(ctx.arrays[i].name, name)) {
            return &ctx.arrays[i];
        }
    }
    return nullptr;
}

static float* f32(const TaskKernelContext& ctx, const char* name) {
    const TaskArrayView* view = find_array(ctx, name);
    return view != nullptr ? static_cast<float*>(view->data) : nullptr;
}

static uint8_t* u8(const TaskKernelContext& ctx, const char* name) {
    const TaskArrayView* view = find_array(ctx, name);
    return view != nullptr ? static_cast<uint8_t*>(view->data) : nullptr;
}

static size_t dim(const TaskKernelContext& ctx, const char* name, size_t axis) {
    const TaskArrayView* view = find_array(ctx, name);
    return view != nullptr && axis < view->rank ? view->shape[axis] : 0;
}

static bool has_array_data(const TaskKernelContext& ctx, const char* name, const void* data) {
    const TaskArrayView* view = find_array(ctx, name);
    if (view == nullptr) {
        return false;
    }
    size_t count = 1;
    for (size_t axis = 0; axis < view->rank; ++axis) {
        count *= view->shape[axis];
    }
    return data != nullptr || count == 0;
}

static inline float param(const float* values, size_t index, float fallback) {
    return values != nullptr ? values[index] : fallback;
}

static inline float flag(const float* values, size_t index) {
    return values != nullptr ? values[index] : 0.0f;
}

static inline float weight(const float* values, size_t index) {
    return values != nullptr ? values[index] : 0.0f;
}
"""


__all__ = [
    "TaskKernel",
    "TaskKernelCompileError",
    "dim",
    "flag",
    "generate_cpp_from_task_kernel",
    "kernel",
    "param",
    "task_kernel",
    "tid",
    "weight",
    "where",
]
