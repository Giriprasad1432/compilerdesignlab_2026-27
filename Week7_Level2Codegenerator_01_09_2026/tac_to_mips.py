"""
Three-Address Code (triples) to MIPS generator -- Level 2 (mixed types).

Supports:
  - INT
  - DOUBLE
  - CHAR
  - STRING
  - RelOpTriple
  - CastTriple
  - SelectTriple (ternary)

Register pools:
  - Integer: $t0-$t9
  - Floating point: $f0, $f2, ..., $f30
"""

from SymbolTable import DataType
from three_address_code import (
    BinOpTriple,
    RelOpTriple,
    CastTriple,
    SelectTriple,
    AssignTriple,
    PrintTriple,
    TripleRef,
)

INT_OP = {
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div',
}

DBL_OP = {
    '+': 'add.d',
    '-': 'sub.d',
    '*': 'mul.d',
    '/': 'div.d',
}

INT_BRANCH_TRUE = {
    '<': 'blt',
    '>': 'bgt',
    '<=': 'ble',
    '>=': 'bge',
    '==': 'beq',
    '!=': 'bne',
}


def literal_kind(operand):
    """
    Returns:
        'int'
        'double'
        'char'
        'string'
        None

    String and char literals retain their quote characters.
    """

    if not isinstance(operand, str):
        return None

    if operand.startswith('"') and operand.endswith('"'):
        return 'string'

    if operand.startswith("'") and operand.endswith("'"):
        return 'char'

    if operand.isdigit():
        return 'int'

    if operand.count('.') == 1:
        a, b = operand.split('.')
        if a.isdigit() and b.isdigit():
            return 'double'

    return None


def triple_result_type(triple):
    """
    Returns the DataType produced by a referenceable triple.
    """

    if isinstance(triple, BinOpTriple):
        return triple.result_type

    if isinstance(triple, RelOpTriple):
        # Comparisons always produce INT 0/1.
        return DataType.INT

    if isinstance(triple, CastTriple):
        return triple.target_type

    if isinstance(triple, SelectTriple):
        return triple.result_type

    raise ValueError(
        f"triple has no referenceable result: {type(triple)}"
    )


class MIPSGenerator:

    def __init__(self, symbol_table, triples):
        self.symbol_table = symbol_table
        self.triples = triples

        # Integer registers:
        # $t0 ... $t9
        self.int_avail = [True] * 10

        # Floating point double-register pairs:
        # $f0/$f1, $f2/$f3, ..., $f30/$f31
        self.float_avail = [True] * 16

        # Triple index -> register
        self.int_index_to_reg = {}
        self.float_index_to_reg = {}

        self.text_lines = []
        self.data_lines = []

        # String literal -> label
        self.string_labels = {}

        self.label_counter = 0

    # ------------------------------------------------------------------
    # Register allocation
    # ------------------------------------------------------------------

    def alloc_int(self):
        i = self.int_avail.index(True)
        self.int_avail[i] = False
        return f"$t{i}"

    def free_int(self, reg):
        i = int(reg[2:])
        self.int_avail[i] = True

    def alloc_float(self):
        i = self.float_avail.index(True)
        self.float_avail[i] = False
        return f"$f{i * 2}"

    def free_float(self, reg):
        i = int(reg[2:]) // 2
        self.float_avail[i] = True

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def new_label(self):
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label

    # ------------------------------------------------------------------
    # String handling
    # ------------------------------------------------------------------

    def get_string_label(self, text):
        """
        Deduplicates identical string literals.
        """

        if text in self.string_labels:
            return self.string_labels[text]

        label = f"Lstr{len(self.string_labels)}"

        self.string_labels[text] = label

        self.data_lines.append(
            f'{label}: .asciiz "{text}"'
        )

        return label

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, line):
        self.text_lines.append(line)

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def is_double_family(self, t):
        return t == DataType.DOUBLE

    def free_reg(self, reg, op_type):
        """
        Free a register from the appropriate register pool.
        """

        if self.is_double_family(op_type):
            self.free_float(reg)
        else:
            self.free_int(reg)

    # ------------------------------------------------------------------
    # Type resolution
    # ------------------------------------------------------------------

    def resolve_type(self, operand):
        """
        Returns the DataType of:
          - literal
          - variable
          - TripleRef
        """

        if isinstance(operand, TripleRef):
            return triple_result_type(
                self.triples[operand.index]
            )

        kind = literal_kind(operand)

        if kind == 'int':
            return DataType.INT

        if kind == 'double':
            return DataType.DOUBLE

        if kind == 'char':
            return DataType.CHAR

        if kind == 'string':
            return DataType.STRING

        entry = self.symbol_table.getSymbol(operand)
        return entry.getDataType()

    # ------------------------------------------------------------------
    # Loading operands
    # ------------------------------------------------------------------

    def load(self, operand, op_type):
        """
        Loads an operand into the appropriate register family.

        Returns:
            (register, was_fresh)

        was_fresh:
            True  -> newly allocated register
            False -> existing TripleRef register reused
        """

        # --------------------------------------------------------------
        # TripleRef
        # --------------------------------------------------------------

        if isinstance(operand, TripleRef):

            if self.is_double_family(op_type):
                return (
                    self.float_index_to_reg[operand.index],
                    False,
                )

            return (
                self.int_index_to_reg[operand.index],
                False,
            )

        # --------------------------------------------------------------
        # Literal
        # --------------------------------------------------------------

        kind = literal_kind(operand)

        # DOUBLE
        if kind == 'double':
            reg = self.alloc_float()

            self.emit(
                f"li.d {reg}, {operand}"
            )

            return reg, True

        # INT
        if kind == 'int':
            reg = self.alloc_int()

            self.emit(
                f"li {reg}, {operand}"
            )

            return reg, True

        # CHAR
        if kind == 'char':
            reg = self.alloc_int()

            char_value = ord(operand[1:-1])

            self.emit(
                f"li {reg}, {char_value}"
            )

            return reg, True

        # STRING
        if kind == 'string':
            reg = self.alloc_int()

            text = operand[1:-1]

            label = self.get_string_label(text)

            self.emit(
                f"la {reg}, {label}"
            )

            return reg, True

        # --------------------------------------------------------------
        # Variable
        # --------------------------------------------------------------

        entry = self.symbol_table.getSymbol(operand)

        offset = entry.getOffset()

        if self.is_double_family(op_type):

            reg = self.alloc_float()

            self.emit(
                f"l.d {reg}, {offset}($fp)"
            )

        else:

            reg = self.alloc_int()

            self.emit(
                f"lw {reg}, {offset}($fp)"
            )

        return reg, True

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_to_var(self, reg, name, reg_type):
        """
        Stores register value into a declared variable.
        """

        entry = self.symbol_table.getSymbol(name)

        offset = entry.getOffset()

        if self.is_double_family(reg_type):

            self.emit(
                f"s.d {reg}, {offset}($fp)"
            )

        else:

            self.emit(
                f"sw {reg}, {offset}($fp)"
            )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def gen_instr(self, triple):

        if isinstance(triple, BinOpTriple):
            self.gen_binop(triple)

        elif isinstance(triple, RelOpTriple):
            self.gen_relop(triple)

        elif isinstance(triple, CastTriple):
            self.gen_cast(triple)

        elif isinstance(triple, SelectTriple):
            self.gen_select(triple)

        elif isinstance(triple, AssignTriple):
            self.gen_assign(triple)

        elif isinstance(triple, PrintTriple):
            self.gen_print(triple)

        else:
            raise ValueError(
                f"unexpected triple type: {type(triple)}"
            )

    # ------------------------------------------------------------------
    # Binary operation
    # ------------------------------------------------------------------

    def gen_binop(self, triple):
        """
        Generate:

            +
            -
            *
            /

        for either INT-family or DOUBLE.
        """

        op_type = triple.result_type

        # Load both operands.
        reg1, fresh1 = self.load(
            triple.arg1,
            op_type
        )

        reg2, fresh2 = self.load(
            triple.arg2,
            op_type
        )

        # DOUBLE
        if self.is_double_family(op_type):

            dest = self.alloc_float()

            instruction = DBL_OP[triple.op]

            self.emit(
                f"{instruction} {dest}, {reg1}, {reg2}"
            )

            self.float_index_to_reg[
                triple.index
            ] = dest

        # INT / CHAR
        else:

            dest = self.alloc_int()

            instruction = INT_OP[triple.op]

            self.emit(
                f"{instruction} {dest}, {reg1}, {reg2}"
            )

            self.int_index_to_reg[
                triple.index
            ] = dest

        # Free freshly loaded operands.
        if fresh1:
            self.free_reg(
                reg1,
                op_type
            )

        if fresh2:
            self.free_reg(
                reg2,
                op_type
            )

    # ------------------------------------------------------------------
    # Relational operation
    # ------------------------------------------------------------------

    def gen_relop(self, triple):
        """
        Generate comparisons.

        INT/CHAR:
            blt/bgt/ble/bge/beq/bne

        DOUBLE:
            c.lt.d
            c.le.d
            c.eq.d
            bc1t/bc1f
        """

        operand_type = triple.operand_type

        reg1, fresh1 = self.load(
            triple.arg1,
            operand_type
        )

        reg2, fresh2 = self.load(
            triple.arg2,
            operand_type
        )

        # Comparison result is ALWAYS INT.
        dest = self.alloc_int()

        true_label = self.new_label()
        end_label = self.new_label()

        # --------------------------------------------------------------
        # DOUBLE comparison
        # --------------------------------------------------------------

        if self.is_double_family(operand_type):

            if triple.op == '<':

                self.emit(
                    f"c.lt.d {reg1}, {reg2}"
                )

                self.emit(
                    f"bc1t {true_label}"
                )

            elif triple.op == '<=':

                self.emit(
                    f"c.le.d {reg1}, {reg2}"
                )

                self.emit(
                    f"bc1t {true_label}"
                )

            elif triple.op == '==':

                self.emit(
                    f"c.eq.d {reg1}, {reg2}"
                )

                self.emit(
                    f"bc1t {true_label}"
                )

            elif triple.op == '>':

                # a > b  ==  b < a
                self.emit(
                    f"c.lt.d {reg2}, {reg1}"
                )

                self.emit(
                    f"bc1t {true_label}"
                )

            elif triple.op == '>=':

                # a >= b  ==  b <= a
                self.emit(
                    f"c.le.d {reg2}, {reg1}"
                )

                self.emit(
                    f"bc1t {true_label}"
                )

            elif triple.op == '!=':

                # Equal flag false means not equal.
                self.emit(
                    f"c.eq.d {reg1}, {reg2}"
                )

                self.emit(
                    f"bc1f {true_label}"
                )

            else:
                raise ValueError(
                    f"unexpected relational operator: "
                    f"{triple.op}"
                )

        # --------------------------------------------------------------
        # INT / CHAR comparison
        # --------------------------------------------------------------

        else:

            branch = INT_BRANCH_TRUE[
                triple.op
            ]

            self.emit(
                f"{branch} {reg1}, {reg2}, {true_label}"
            )

        # --------------------------------------------------------------
        # False result
        # --------------------------------------------------------------

        self.emit(
            f"li {dest}, 0"
        )

        self.emit(
            f"b {end_label}"
        )

        # --------------------------------------------------------------
        # True result
        # --------------------------------------------------------------

        self.emit(
            f"{true_label}:"
        )

        self.emit(
            f"li {dest}, 1"
        )

        self.emit(
            f"{end_label}:"
        )

        # Comparison produces INT.
        self.int_index_to_reg[
            triple.index
        ] = dest

        # Free freshly loaded operands.
        if fresh1:
            self.free_reg(
                reg1,
                operand_type
            )

        if fresh2:
            self.free_reg(
                reg2,
                operand_type
            )

    # ------------------------------------------------------------------
    # Cast
    # ------------------------------------------------------------------

    def gen_cast(self, triple):
        """
        Generate:

            DOUBLE -> INT/CHAR
            INT/CHAR -> DOUBLE
            same-family cast
        """

        src, fresh = self.load(
            triple.arg,
            triple.source_type
        )

        source_double = self.is_double_family(
            triple.source_type
        )

        target_double = self.is_double_family(
            triple.target_type
        )

        # --------------------------------------------------------------
        # DOUBLE -> INT/CHAR
        # --------------------------------------------------------------

        if source_double and not target_double:

            # Scratch FPU register.
            tmp = self.alloc_float()

            self.emit(
                f"cvt.w.d {tmp}, {src}"
            )

            # Move converted integer out of FPU.
            dest = self.alloc_int()

            self.emit(
                f"mfc1 {dest}, {tmp}"
            )

            self.free_float(tmp)

            self.int_index_to_reg[
                triple.index
            ] = dest

            if fresh:
                self.free_float(src)

        # --------------------------------------------------------------
        # INT/CHAR -> DOUBLE
        # --------------------------------------------------------------

        elif not source_double and target_double:

            # Scratch FPU register.
            tmp = self.alloc_float()

            self.emit(
                f"mtc1 {src}, {tmp}"
            )

            dest = self.alloc_float()

            self.emit(
                f"cvt.d.w {dest}, {tmp}"
            )

            self.free_float(tmp)

            self.float_index_to_reg[
                triple.index
            ] = dest

            if fresh:
                self.free_int(src)

        # --------------------------------------------------------------
        # Same register family
        # --------------------------------------------------------------

        else:

            # CHAR <-> INT requires no actual instruction.
            #
            # The existing register contains the correct bit pattern,
            # so this triple simply aliases it.
            if target_double:

                self.float_index_to_reg[
                    triple.index
                ] = src

            else:

                self.int_index_to_reg[
                    triple.index
                ] = src

            # DO NOT FREE src.
            #
            # It is now the result of this triple.

    # ------------------------------------------------------------------
    # Ternary / Select
    # ------------------------------------------------------------------

    def gen_select(self, triple):
        """
        Generate:

            cond ? then_val : else_val

        using branches.

        Destination is allocated BEFORE loading the condition or either
        branch operand.
        """

        result_type = triple.result_type

        is_double = self.is_double_family(
            result_type
        )

        # --------------------------------------------------------------
        # Allocate destination FIRST.
        # --------------------------------------------------------------

        if is_double:
            dest = self.alloc_float()
        else:
            dest = self.alloc_int()

        # --------------------------------------------------------------
        # Condition
        # --------------------------------------------------------------

        cond, fresh_cond = self.load(
            triple.cond,
            triple.cond_type
        )

        then_label = self.new_label()
        end_label = self.new_label()

        # --------------------------------------------------------------
        # Test condition
        # --------------------------------------------------------------

        if self.is_double_family(
            triple.cond_type
        ):

            # A double is truthy when it is nonzero.
            #
            # Compare cond == 0.
            # If comparison is FALSE, cond != 0,
            # therefore branch to THEN.

            zero = self.alloc_float()

            self.emit(
                f"li.d {zero}, 0.0"
            )

            self.emit(
                f"c.eq.d {cond}, {zero}"
            )

            self.emit(
                f"bc1f {then_label}"
            )

            self.free_float(zero)

        else:

            # INT/CHAR truthiness:
            # nonzero means true.
            self.emit(
                f"bne {cond}, $zero, {then_label}"
            )

        # Condition is no longer needed after the branch.
        if fresh_cond:
            self.free_reg(
                cond,
                triple.cond_type
            )

        # --------------------------------------------------------------
        # ELSE branch
        # --------------------------------------------------------------

        else_reg, fresh_else = self.load(
            triple.else_val,
            result_type
        )

        if is_double:

            self.emit(
                f"mov.d {dest}, {else_reg}"
            )

        else:

            self.emit(
                f"move {dest}, {else_reg}"
            )

        if fresh_else:
            self.free_reg(
                else_reg,
                result_type
            )

        # Skip THEN branch.
        self.emit(
            f"b {end_label}"
        )

        # --------------------------------------------------------------
        # THEN branch
        # --------------------------------------------------------------

        self.emit(
            f"{then_label}:"
        )

        then_reg, fresh_then = self.load(
            triple.then_val,
            result_type
        )

        if is_double:

            self.emit(
                f"mov.d {dest}, {then_reg}"
            )

        else:

            self.emit(
                f"move {dest}, {then_reg}"
            )

        if fresh_then:
            self.free_reg(
                then_reg,
                result_type
            )

        # --------------------------------------------------------------
        # End
        # --------------------------------------------------------------

        self.emit(
            f"{end_label}:"
        )

        # Store destination as this triple's result.
        if is_double:

            self.float_index_to_reg[
                triple.index
            ] = dest

        else:

            self.int_index_to_reg[
                triple.index
            ] = dest

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def gen_assign(self, triple):
        """
        Assignment:

            variable = expression
        """

        dest_type = (
            self.symbol_table
            .getSymbol(triple.dest)
            .getDataType()
        )

        reg, fresh = self.load(
            triple.arg1,
            dest_type
        )

        self.store_to_var(
            reg,
            triple.dest,
            dest_type
        )

        # Only a freshly-loaded register belongs to this assignment.
        #
        # A reused TripleRef register remains allocated because a later
        # triple may still reference its producing triple.
        if fresh:
            self.free_reg(
                reg,
                dest_type
            )

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    def gen_print(self, triple):
        """
        Print:
            INT    -> syscall 1
            DOUBLE -> syscall 3
            STRING -> syscall 4
            CHAR   -> syscall 11
        """

        value_type = self.resolve_type(
            triple.arg1
        )

        reg, fresh = self.load(
            triple.arg1,
            value_type
        )

        # --------------------------------------------------------------
        # DOUBLE
        # --------------------------------------------------------------

        if value_type == DataType.DOUBLE:

            self.emit(
                f"mov.d $f12, {reg}"
            )

            self.emit(
                "li $v0, 3"
            )

            self.emit(
                "syscall"
            )

        # --------------------------------------------------------------
        # STRING
        # --------------------------------------------------------------

        elif value_type == DataType.STRING:

            self.emit(
                f"move $a0, {reg}"
            )

            self.emit(
                "li $v0, 4"
            )

            self.emit(
                "syscall"
            )

        # --------------------------------------------------------------
        # CHAR
        # --------------------------------------------------------------

        elif value_type == DataType.CHAR:

            self.emit(
                f"move $a0, {reg}"
            )

            self.emit(
                "li $v0, 11"
            )

            self.emit(
                "syscall"
            )

        # --------------------------------------------------------------
        # INT
        # --------------------------------------------------------------

        else:

            self.emit(
                f"move $a0, {reg}"
            )

            self.emit(
                "li $v0, 1"
            )

            self.emit(
                "syscall"
            )

        if fresh:
            self.free_reg(
                reg,
                value_type
            )

    # ------------------------------------------------------------------
    # Prologue
    # ------------------------------------------------------------------

    def emit_prologue(self, frame_size):

        self.emit(
            "subu $sp, $sp, 4"
        )

        self.emit(
            "sw   $ra, 0($sp)"
        )

        self.emit(
            "subu $sp, $sp, 4"
        )

        self.emit(
            "sw   $fp, 0($sp)"
        )

        self.emit(
            f"addiu $fp, $sp, -{frame_size}"
        )

        self.emit(
            "move $sp, $fp"
        )

    # ------------------------------------------------------------------
    # Epilogue
    # ------------------------------------------------------------------

    def emit_epilogue(self, frame_size):

        self.emit(
            f"addiu $sp, $fp, {frame_size}"
        )

        self.emit(
            "lw    $fp, 0($sp)"
        )

        self.emit(
            "addiu $sp, $sp, 4"
        )

        self.emit(
            "lw    $ra, 0($sp)"
        )

        self.emit(
            "addiu $sp, $sp, 4"
        )

        self.emit(
            "jr    $ra"
        )

    # ------------------------------------------------------------------
    # Generate complete program
    # ------------------------------------------------------------------

    def generate(self, triples, frame_size):

        self.emit_prologue(frame_size)

        for triple in triples:
            self.gen_instr(triple)

        self.emit_epilogue(frame_size)

        return self.render()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):

        lines = []

        # .data is only emitted if strings were encountered.
        if self.data_lines:

            lines.append(".data")
            lines.extend(self.data_lines)

        lines.append(".text")
        lines.append(".globl main")
        lines.append("main:")

        lines.extend(
            f"    {line}"
            for line in self.text_lines
        )

        return "\n".join(lines) + "\n"


# ======================================================================
# Convenience wrapper
# ======================================================================

def generate_mips(function, triple_program):
    """
    Convenience wrapper.

    Assumes assignOffsetsToSymbols() has already been called on
    function.getLocalSymbolTable().
    """

    gen = MIPSGenerator(
        function.getLocalSymbolTable(),
        triple_program.triples
    )

    return gen.generate(
        triple_program,
        function.getLocalSymbolTable().size()
    )

