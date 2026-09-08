from ast_nodes import Const, Var, Assign, Print, BinOp, RelOp, Cast, Ternary
from SymbolTable import DataType
from type_rules import is_numeric, promote, SemanticError


class TypeChecker:

    def __init__(self, symbol_table):
        self.symbol_table = symbol_table
        self.errors = []

    def error(self, message, lineno):
        self.errors.append(SemanticError(message, lineno))

    def check(self, node):

        if isinstance(node, Const):
            return (node, node.type)

        elif isinstance(node, Var):
            return self.check_var(node)

        elif isinstance(node, Assign):
            return self.check_assign_stmt(node)

        elif isinstance(node, Print):
            return self.check_print(node)

        elif isinstance(node, BinOp):
            return self.check_binop(node)

        elif isinstance(node, RelOp):
            return self.check_relop(node)

        elif isinstance(node, Cast):
            return self.check_cast(node)

        elif isinstance(node, Ternary):
            return self.check_ternary(node)

        return (node, DataType.INT)

    def check_var(self, node):

        entry = self.symbol_table.getSymbol(node.name)

        if entry:
            return (node, entry.getDataType())

        self.error(
            f"undeclared variable '{node.name}'",
            node.lineno
        )

        return (node, DataType.INT)

    def check_binop(self, node):

        new_left, left_type = self.check(node.left)
        new_right, right_type = self.check(node.right)

        node.left = new_left
        node.right = new_right

        if not is_numeric(left_type) or not is_numeric(right_type):

            self.error(
                f"incompatible types for '{node.op}': "
                f"{left_type.name} and {right_type.name}",
                node.lineno
            )

            return (node, DataType.INT)

        result_type = promote(left_type, right_type)

        if left_type != result_type:
            node.left = Cast(
                result_type,
                node.left,
                lineno=node.left.lineno
            )

        if right_type != result_type:
            node.right = Cast(
                result_type,
                node.right,
                lineno=node.right.lineno
            )

        return (node, result_type)

    def check_relop(self, node):

        new_left, left_type = self.check(node.left)
        new_right, right_type = self.check(node.right)

        node.left = new_left
        node.right = new_right

        if is_numeric(left_type) and is_numeric(right_type):

            common_type = promote(left_type, right_type)

            if left_type != common_type:
                node.left = Cast(
                    common_type,
                    node.left,
                    lineno=node.left.lineno
                )

            if right_type != common_type:
                node.right = Cast(
                    common_type,
                    node.right,
                    lineno=node.right.lineno
                )

            return (node, DataType.INT)

        elif left_type == DataType.STRING and right_type == DataType.STRING:

            return (node, DataType.INT)

        else:

            self.error(
                f"incompatible types for '{node.op}': "
                f"{left_type.name} and {right_type.name}",
                node.lineno
            )

            return (node, DataType.INT)

    def check_ternary(self, node):

        new_cond, cond_type = self.check(node.cond)
        new_then, then_type = self.check(node.then_expr)
        new_else, else_type = self.check(node.else_expr)

        node.cond = new_cond
        node.then_expr = new_then
        node.else_expr = new_else

        if not is_numeric(cond_type):

            self.error(
                f"ternary condition must be numeric, "
                f"got {cond_type.name}",
                node.cond.lineno
            )

        if is_numeric(then_type) and is_numeric(else_type):

            result_type = promote(then_type, else_type)

            if then_type != result_type:
                node.then_expr = Cast(
                    result_type,
                    node.then_expr,
                    lineno=node.then_expr.lineno
                )

            if else_type != result_type:
                node.else_expr = Cast(
                    result_type,
                    node.else_expr,
                    lineno=node.else_expr.lineno
                )

            return (node, result_type)

        if then_type == else_type:
            return (node, then_type)

        self.error(
            f"incompatible ternary branches: "
            f"{then_type.name} and {else_type.name}",
            node.lineno
        )

        return (node, then_type)

    def check_cast(self, node):

        new_expr, expr_type = self.check(node.expr)

        node.expr = new_expr

        target_type = node.target_type

        if is_numeric(expr_type) and is_numeric(target_type):
            return (node, target_type)

        if expr_type == target_type:
            return (node, target_type)

        self.error(
            f"invalid cast from {expr_type.name} "
            f"to {target_type.name}",
            node.lineno
        )

        return (node, target_type)

    def check_assign_stmt(self, node):

        new_var, var_type = self.check_var(node.var)

        new_expr, expr_type = self.check(node.expr)

        node.var = new_var
        node.expr = new_expr

        if var_type == expr_type:
            return (node, var_type)

        if is_numeric(var_type) and is_numeric(expr_type):

            promoted_type = promote(var_type, expr_type)

            if promoted_type == var_type:

                node.expr = Cast(
                    var_type,
                    node.expr,
                    lineno=node.expr.lineno
                )

                return (node, var_type)

        self.error(
            f"cannot assign {expr_type.name} to {var_type.name}",
            node.lineno
        )

        return (node, var_type)

    def check_print(self, node):

        new_expr, expr_type = self.check(node.expr)

        node.expr = new_expr

        return (node, expr_type)


def check_program(program):

    errors = []

    for func in program.getFunctions():

        symbol_table = func.getLocalSymbolTable()

        checker = TypeChecker(symbol_table)

        for stmt in func.getStatementsAstList():
            checker.check(stmt)

        errors.extend(checker.errors)

    return errors

