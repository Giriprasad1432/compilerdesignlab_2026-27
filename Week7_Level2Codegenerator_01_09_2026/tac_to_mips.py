"""
Three-Address Code (triples) to MIPS generator -- Level 2 (mixed types).

Read docs/typed_3ac_reference.md and docs/mips_fp_reference.md before
editing this file.

WEEK 7: this extends Week 4's single-register-family (integer-only)
generator to handle DOUBLE, CHAR, and STRING alongside INT, and adds
codegen for RelOp/Cast/Ternary (via the RelOpTriple/CastTriple/
SelectTriple triples tac_generator.py now emits).

Two SEPARATE register pools now, tracked independently:
  - Integer: $t0-$t9 (unchanged from Week 4)
  - Floating-point: $f0, $f2, $f4, ..., $f30 -- MIPS's FPU registers are
    used in EVEN-numbered PAIRS to hold one 8-byte double each (the odd
    register of each pair holds the high-order half). This code always
    allocates/frees by the even index, so you never need to think about
    the paired odd register directly -- see docs/mips_fp_reference.md.

Every triple's operand(s) get resolved to the RIGHT pool based on their
TYPE (looked up via resolve_type() / literal_kind(), both provided) --
an int-family value always goes through alloc_int()/the integer
instructions; a double-family value always goes through
alloc_float()/the ".d" instructions.
"""
from SymbolTable import DataType
from three_address_code import (
    BinOpTriple, RelOpTriple, CastTriple, SelectTriple, AssignTriple, PrintTriple,
    TripleRef, is_literal,
)

INT_OP = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
DBL_OP = {'+': 'add.d', '-': 'sub.d', '*': 'mul.d', '/': 'div.d'}

# int comparison -> branch-if-true pseudo-instruction (SPIM provides
# blt/bgt/ble/bge/beq/bne as 3-operand pseudo-ops -- verified directly
# on real SPIM, see docs/mips_fp_reference.md).
INT_BRANCH_TRUE = {'<': 'blt', '>': 'bgt', '<=': 'ble', '>=': 'bge', '==': 'beq', '!=': 'bne'}


def literal_kind(operand):
    """
    Provided. Returns 'int' / 'double' / 'char' / 'string' for a literal
    operand, or None if `operand` is a variable name instead. String and
    char literals are recognizable because tac_generator.py's Const
    handling wraps them in their original quote characters
    ('"hello"' / "'x'") specifically so this function can tell them
    apart from a bare variable name by inspection -- see
    docs/typed_3ac_reference.md.
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
    Provided. Returns the DataType a given (already-generated) triple's
    result has, for resolving a TripleRef operand's type. Only triples
    that PRODUCE a referenceable value need a case here (AssignTriple/
    PrintTriple never do -- nothing ever builds a TripleRef to one).
    """
    if isinstance(triple, BinOpTriple):
        return triple.result_type
    if isinstance(triple, RelOpTriple):
        return DataType.INT  # a comparison's own value is always INT 0/1
    if isinstance(triple, CastTriple):
        return triple.target_type
    if isinstance(triple, SelectTriple):
        return triple.result_type
    raise ValueError(f"triple has no referenceable result: {type(triple)}")


class MIPSGenerator:
    def __init__(self, symbol_table, triples):
        self.symbol_table = symbol_table
        self.triples = triples  # needed to resolve a TripleRef operand's type
        self.int_avail = [True] * 10
        self.float_avail = [True] * 16  # represents $f0,$f2,...,$f30
        self.int_index_to_reg = {}
        self.float_index_to_reg = {}
        self.text_lines = []
        self.data_lines = []
        self.string_labels = {}
        self.label_counter = 0

    # ------------------------------------------------------------------
    # Register allocation, labels, string data -- PROVIDED
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

    def new_label(self):
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label

    def get_string_label(self, text):
        """Deduplicates identical string literals to one .data entry each."""
        if text in self.string_labels:
            return self.string_labels[text]
        label = f"Lstr{len(self.string_labels)}"
        self.string_labels[text] = label
        self.data_lines.append(f'{label}: .asciiz "{text}"')
        return label

    def emit(self, line):
        self.text_lines.append(line)

    def is_double_family(self, t):
        return t == DataType.DOUBLE

    def free_reg(self, reg, op_type):
        """Frees `reg` from whichever pool matches op_type."""
        if self.is_double_family(op_type):
            self.free_float(reg)
        else:
            self.free_int(reg)

    # ------------------------------------------------------------------
    # Type resolution -- PROVIDED
    # ------------------------------------------------------------------
    def resolve_type(self, operand):
        """
        Returns the DataType of ANY operand (literal, variable name, or
        TripleRef). Used by gen_print() and by load()'s callers that
        don't already know the type from context.
        """
        if isinstance(operand, TripleRef):
            return triple_result_type(self.triples[operand.index])
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
    # Loading/storing operands -- TODO
    # ------------------------------------------------------------------
    def load(self, operand, op_type):
        """
        TODO(week-7): get `operand`'s value into a register of the
        family matching op_type, and return (reg, was_fresh) --
        was_fresh is False only for the TripleRef-reuse case (mirrors
        Week 4's convention: a reused register must not be freed by the
        caller, a freshly-loaded one should be).

          isinstance(operand, TripleRef):
              reuse self.float_index_to_reg[operand.index] if
              is_double_family(op_type) else
              self.int_index_to_reg[operand.index] -- return (that_reg, False)

          literal_kind(operand) == 'double':
              reg = self.alloc_float(); emit `li.d reg, operand`

          literal_kind(operand) == 'int':
              reg = self.alloc_int(); emit `li reg, operand`

          literal_kind(operand) == 'char':
              reg = self.alloc_int(); emit `li reg, {ord(operand[1:-1])}`
              -- operand is like "'x'" (with quote characters still on
              it, see literal_kind()'s docstring), so operand[1:-1]
              strips them before ord() converts to the character code.

          literal_kind(operand) == 'string':
              reg = self.alloc_int()
              label = self.get_string_label(operand[1:-1])  # strip quotes
              emit `la reg, label`  -- NOT li; `la` loads an ADDRESS

          otherwise (a variable name):
              entry = self.symbol_table.getSymbol(operand)
              offset = entry.getOffset()
              if is_double_family(op_type): reg = alloc_float(); emit `l.d reg, offset($fp)`
              else: reg = alloc_int(); emit `lw reg, offset($fp)`

        In every non-TripleRef case, return (reg, True).
        """
        if isinstance(operand, TripleRef):
            if self.is_double_family(op_type):
                return self.float_index_to_reg[operand.index], False
            return self.int_index_to_reg[operand.index], False

        kind = literal_kind(operand)

        if kind == 'double':
            reg = self.alloc_float()
            self.emit(f"li.d {reg}, {operand}")
            return reg, True

        if kind == 'int':
            reg = self.alloc_int()
            self.emit(f"li {reg}, {operand}")
            return reg, True

        if kind == 'char':
            reg = self.alloc_int()
            self.emit(f"li {reg}, {ord(operand[1:-1])}")
            return reg, True

        if kind == 'string':
            reg = self.alloc_int()
            label = self.get_string_label(operand[1:-1])
            self.emit(f"la {reg}, {label}")
            return reg, True

        entry = self.symbol_table.getSymbol(operand)
        offset = entry.getOffset()

        if self.is_double_family(op_type):
            reg = self.alloc_float()
            self.emit(f"l.d {reg}, {offset}($fp)")
        else:
            reg = self.alloc_int()
            self.emit(f"lw {reg}, {offset}($fp)")

        return reg, True

    def store_to_var(self, reg, name, reg_type):
        """
        TODO(week-7): look up name's offset via self.symbol_table, then
        emit `s.d reg, offset($fp)` if is_double_family(reg_type), else
        `sw reg, offset($fp)`.
        """
        entry = self.symbol_table.getSymbol(name)
        offset = entry.getOffset()

        if self.is_double_family(reg_type):
            self.emit(f"s.d {reg}, {offset}($fp)")
        else:
            self.emit(f"sw {reg}, {offset}($fp)")

    # ------------------------------------------------------------------
    # Per-triple codegen -- TODO (this is the bulk of the week)
    # ------------------------------------------------------------------
    def gen_instr(self, triple):
        """
        Provided -- dispatches to the right gen_* method. You should not
        need to change this method.
        """
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
            raise ValueError(f"unexpected triple type: {type(triple)}")

    def gen_binop(self, triple):
        """
        TODO(week-7): load both operands using triple.result_type
        (Week 6 already ensured arg1/arg2 are the same, already-promoted
        type as the result -- see docs/typed_3ac_reference.md). Pick
        INT_OP or DBL_OP based on is_double_family(triple.result_type),
        allocate a destination register from the matching pool, emit the
        instruction, record the destination in int_index_to_reg or
        float_index_to_reg (keyed by triple.index) accordingly, and free
        any operand register that was freshly loaded (mirroring Week 4's
        register-reuse-via-TripleRef convention exactly, just now across
        two separate pools instead of one).
        """
        op_type = triple.result_type
        left, left_fresh = self.load(triple.arg1, op_type)
        right, right_fresh = self.load(triple.arg2, op_type)

        if self.is_double_family(op_type):
            dest = self.alloc_float()
            instruction = DBL_OP[triple.op]
            self.emit(f"{instruction} {dest}, {left}, {right}")
            self.float_index_to_reg[triple.index] = dest
        else:
            dest = self.alloc_int()
            instruction = INT_OP[triple.op]
            self.emit(f"{instruction} {dest}, {left}, {right}")
            self.int_index_to_reg[triple.index] = dest

        if left_fresh:
            self.free_reg(left, op_type)
        if right_fresh:
            self.free_reg(right, op_type)

    def gen_relop(self, triple):
        """
        TODO(week-7): implement using the UNIFORM branch-based pattern
        (same shape for int and double operands, just different
        condition-testing instructions -- see
        docs/mips_fp_reference.md for why, and the exact instruction
        tables to use):

          1. Load arg1/arg2 using triple.operand_type (NOT
             triple's own result type -- a RelOpTriple's operands can be
             DOUBLE while its own produced value is always INT).
          2. Allocate an INT destination register (comparisons always
             produce an INT 0/1, regardless of operand type).
          3. Emit the condition test + branch:
             - if is_double_family(triple.operand_type): use the
               c.<cond>.d + bc1t/bc1f pattern from
               docs/mips_fp_reference.md's comparison table (note '>' and
               '>=' need their operands SWAPPED, and '!=' needs bc1f
               instead of bc1t -- there's no direct "not equal" or
               "greater than" FPU compare instruction).
             - otherwise: emit INT_BRANCH_TRUE[triple.op] as a 3-operand
               branch pseudo-instruction directly (no compare instruction
               needed first -- unlike the double case).
          4. Emit the standard "materialize 0 or 1" skeleton:
                 li dest, 0
                 b Lend
                 Ltrue:
                 li dest, 1
                 Lend:
             (two fresh labels from self.new_label())
          5. Record dest in int_index_to_reg[triple.index]; free any
             freshly-loaded operand registers.
        """
        op_type = triple.operand_type
        left, left_fresh = self.load(triple.arg1, op_type)
        right, right_fresh = self.load(triple.arg2, op_type)

        dest = self.alloc_int()
        l_true = self.new_label()
        l_end = self.new_label()

        if self.is_double_family(op_type):
            if triple.op == '<':
                self.emit(f"c.lt.d {left}, {right}")
                self.emit(f"bc1t {l_true}")
            elif triple.op == '>':
                self.emit(f"c.lt.d {right}, {left}")
                self.emit(f"bc1t {l_true}")
            elif triple.op == '<=':
                self.emit(f"c.le.d {left}, {right}")
                self.emit(f"bc1t {l_true}")
            elif triple.op == '>=':
                self.emit(f"c.le.d {right}, {left}")
                self.emit(f"bc1t {l_true}")
            elif triple.op == '==':
                self.emit(f"c.eq.d {left}, {right}")
                self.emit(f"bc1t {l_true}")
            elif triple.op == '!=':
                self.emit(f"c.eq.d {left}, {right}")
                self.emit(f"bc1f {l_true}")
            else:
                raise ValueError(f"unknown relational operator: {triple.op}")
        else:
            branch = INT_BRANCH_TRUE[triple.op]
            self.emit(f"{branch} {left}, {right}, {l_true}")

        self.emit(f"li {dest}, 0")
        self.emit(f"b {l_end}")
        self.emit(f"{l_true}:")
        self.emit(f"li {dest}, 1")
        self.emit(f"{l_end}:")

        self.int_index_to_reg[triple.index] = dest

        if left_fresh:
            self.free_reg(left, op_type)
        if right_fresh:
            self.free_reg(right, op_type)

    def gen_cast(self, triple):
        """
        TODO(week-7): load triple.arg using triple.source_type. Three
        cases, based on triple.source_type/triple.target_type:

          source DOUBLE, target INT (or CHAR):
              tmp = self.alloc_float()
              emit `cvt.w.d tmp, src`
              dest = self.alloc_int()
              emit `mfc1 dest, tmp`
              self.free_float(tmp)
              record dest in int_index_to_reg[triple.index]

          source INT (or CHAR), target DOUBLE:
              tmp = self.alloc_float()
              emit `mtc1 src, tmp`
              dest = self.alloc_float()
              emit `cvt.d.w dest, tmp`
              self.free_float(tmp)
              record dest in float_index_to_reg[triple.index]

          same family both sides (e.g. CHAR<->INT): no conversion
          instruction needed at all -- the loaded register IS already
          the right bit pattern. Just record src directly as this
          triple's result (in whichever dict matches target_type) and
          do NOT free it, even if it was freshly loaded -- it's still in
          use as the cast's own result now.

        Free the ORIGINAL loaded src register only in the two
        conversion cases above, and only if it was freshly loaded (not a
        reused TripleRef).
        """
        src, src_fresh = self.load(triple.arg, triple.source_type)

        source_double = self.is_double_family(triple.source_type)
        target_double = self.is_double_family(triple.target_type)

        if source_double and not target_double:
            tmp = self.alloc_float()
            self.emit(f"cvt.w.d {tmp}, {src}")

            dest = self.alloc_int()
            self.emit(f"mfc1 {dest}, {tmp}")
            self.free_float(tmp)

            self.int_index_to_reg[triple.index] = dest

            if src_fresh:
                self.free_float(src)

        elif not source_double and target_double:
            tmp = self.alloc_float()
            self.emit(f"mtc1 {src}, {tmp}")

            dest = self.alloc_float()
            self.emit(f"cvt.d.w {dest}, {tmp}")
            self.free_float(tmp)

            self.float_index_to_reg[triple.index] = dest

            if src_fresh:
                self.free_int(src)

        else:
            if target_double:
                self.float_index_to_reg[triple.index] = src
            else:
                self.int_index_to_reg[triple.index] = src
            # src becomes the result of this cast, so it must stay allocated.

    def gen_select(self, triple):
        """
        TODO(week-7): the ternary operator, entirely branch-based (this
        is the specific instruction from this week's lab plan). The
        tricky part: BOTH branches must end up writing their value into
        the SAME destination register, since there's no way to know
        which branch ran by the time execution reaches the code after
        the ternary. See docs/mips_fp_reference.md for a fully worked
        trace of this exact pattern.

        Order matters -- allocate dest FIRST, before evaluating either
        branch, so the allocator can't hand out dest's register to
        anything used while computing the branches:

          1. dest = alloc_float() or alloc_int(), based on
             is_double_family(triple.result_type).
          2. Load triple.cond using triple.cond_type. Test its
             truthiness and branch to a "then" label if true:
             - double cond: compare against a zero register via c.eq.d,
               branch on bc1f (branch when NOT equal to zero, i.e. when
               truthy) -- see docs/mips_fp_reference.md.
             - int/char cond: `bne cond_reg, $zero, Lthen`
             Free cond_reg if it was freshly loaded.
          3. ELSE branch: load triple.else_val using triple.result_type,
             move it into dest (`mov.d dest, reg` or `move dest, reg`
             depending on family), free the loaded register if fresh,
             then unconditionally branch to an "end" label.
          4. THEN label, then: load triple.then_val using
             triple.result_type, move it into dest the same way, free if
             fresh.
          5. END label.
          6. Record dest in int_index_to_reg[triple.index] or
             float_index_to_reg[triple.index], matching triple.result_type.
        """
        result_type = triple.result_type

        if self.is_double_family(result_type):
            dest = self.alloc_float()
        else:
            dest = self.alloc_int()

        l_then = self.new_label()
        l_end = self.new_label()

        cond, cond_fresh = self.load(triple.cond, triple.cond_type)

        if self.is_double_family(triple.cond_type):
            zero = self.alloc_float()
            self.emit(f"li.d {zero}, 0.0")
            self.emit(f"c.eq.d {cond}, {zero}")
            self.emit(f"bc1f {l_then}")
            self.free_float(zero)
        else:
            self.emit(f"bne {cond}, $zero, {l_then}")

        if cond_fresh:
            self.free_reg(cond, triple.cond_type)

        else_reg, else_fresh = self.load(triple.else_val, result_type)
        if self.is_double_family(result_type):
            self.emit(f"mov.d {dest}, {else_reg}")
        else:
            self.emit(f"move {dest}, {else_reg}")

        if else_fresh:
            self.free_reg(else_reg, result_type)

        self.emit(f"b {l_end}")
        self.emit(f"{l_then}:")

        then_reg, then_fresh = self.load(triple.then_val, result_type)
        if self.is_double_family(result_type):
            self.emit(f"mov.d {dest}, {then_reg}")
        else:
            self.emit(f"move {dest}, {then_reg}")

        if then_fresh:
            self.free_reg(then_reg, result_type)

        self.emit(f"{l_end}:")

        if self.is_double_family(result_type):
            self.float_index_to_reg[triple.index] = dest
        else:
            self.int_index_to_reg[triple.index] = dest

    def gen_assign(self, triple):
        """
        TODO(week-7): look up triple.dest's declared type via
        self.symbol_table.getSymbol(triple.dest).getDataType(), load()
        triple.arg1 using that type, store_to_var() the result, then
        free the register (mirroring Week 4's "store always frees its
        register" convention -- true regardless of which pool the
        register came from).
        """
        dest_type = self.symbol_table.getSymbol(triple.dest).getDataType()
        reg, fresh = self.load(triple.arg1, dest_type)
        self.store_to_var(reg, triple.dest, dest_type)

        if fresh:
            self.free_reg(reg, dest_type)

    def gen_print(self, triple):
        """
        TODO(week-7): resolve_type(triple.arg1) to find what's being
        printed, load() it, then pick the syscall by type:

          DataType.DOUBLE: `mov.d $f12, reg` / `li $v0, 3` / `syscall`
          DataType.STRING: `move $a0, reg` / `li $v0, 4` / `syscall`
          DataType.CHAR:   `move $a0, reg` / `li $v0, 11` / `syscall`
                           (print-character syscall -- not in the
                           lab plan's own list, added here so `print c;`
                           for a char variable shows the actual
                           character rather than its ordinal number;
                           see docs/mips_fp_reference.md)
          otherwise (INT): `move $a0, reg` / `li $v0, 1` / `syscall`
                           (unchanged from Week 4)

        Free the register afterward if it was freshly loaded.
        """
        op_type = self.resolve_type(triple.arg1)
        reg, fresh = self.load(triple.arg1, op_type)

        if op_type == DataType.DOUBLE:
            self.emit(f"mov.d $f12, {reg}")
            self.emit("li $v0, 3")
            self.emit("syscall")
        elif op_type == DataType.STRING:
            self.emit(f"move $a0, {reg}")
            self.emit("li $v0, 4")
            self.emit("syscall")
        elif op_type == DataType.CHAR:
            self.emit(f"move $a0, {reg}")
            self.emit("li $v0, 11")
            self.emit("syscall")
        else:
            self.emit(f"move $a0, {reg}")
            self.emit("li $v0, 1")
            self.emit("syscall")

        if fresh:
            self.free_reg(reg, op_type)

    # ------------------------------------------------------------------
    # Prologue / epilogue -- PROVIDED, unchanged from Week 4
    # ------------------------------------------------------------------
    def emit_prologue(self, frame_size):
        self.emit("subu $sp, $sp, 4")
        self.emit("sw   $ra, 0($sp)")
        self.emit("subu $sp, $sp, 4")
        self.emit("sw   $fp, 0($sp)")
        self.emit(f"addiu $fp, $sp, -{frame_size}")
        self.emit("move $sp, $fp")

    def emit_epilogue(self, frame_size):
        self.emit(f"addiu $sp, $fp, {frame_size}")
        self.emit("lw    $fp, 0($sp)")
        self.emit("addiu $sp, $sp, 4")
        self.emit("lw    $ra, 0($sp)")
        self.emit("addiu $sp, $sp, 4")
        self.emit("jr    $ra")

    def generate(self, triples, frame_size):
        """
        Provided. Emits the prologue, the body (one gen_instr() call per
        triple, in order), then the epilogue, and renders the final .s
        text. You should not need to change this method.
        """
        self.emit_prologue(frame_size)
        for triple in triples:
            self.gen_instr(triple)
        self.emit_epilogue(frame_size)
        return self.render()

    def render(self):
        """Provided. Assembles .data (only if string literals were used) + .text."""
        lines = []
        if self.data_lines:
            lines.append(".data")
            lines.extend(self.data_lines)
        lines.append(".text")
        lines.append(".globl main")
        lines.append("main:")
        lines.extend(f"    {line}" for line in self.text_lines)
        return "\n".join(lines) + "\n"


def generate_mips(function, triple_program):
    """
    Convenience wrapper. Assumes assignOffsetsToSymbols() has already
    been called on function.getLocalSymbolTable() -- see Function.compile().
    """
    gen = MIPSGenerator(function.getLocalSymbolTable(), triple_program.triples)
    return gen.generate(triple_program, function.getLocalSymbolTable().size())
