from sly import Parser
from ExprAstLexer import ExprAstLexer
from ast_nodes import * 
from tac_generator import *
from three_address_code import render_threeAddressCode 
from tac_to_mips import generate_mips

class ExprAstParser(Parser):
    tokens = ExprAstLexer.tokens
    
    # A -> id = E
    @_('ID "=" E')
    def A(self, value):
        return Assign(Var(value[0]), value[2])

    # E -> E + T
    @_('E "+" T')
    def E(self, value):
        return BinOp('+', value[0], value[2])

    # E -> E - T
    @_('E "-" T')
    def E(self, value):
        return BinOp('-', value[0], value[2])

    # E -> T
    @_('T')
    def E(self, value):
        return value[0]

    # T -> T * F
    @_('T "*" F')
    def T(self, value):
        return BinOp('*', value[0], value[2])

    # T -> T / F
    @_('T "/" F')
    def T(self, value):
        return BinOp('/', value[0], value[2])

    # T -> T % F
    @_('T "%" F')
    def T(self, value):
        return BinOp('%', value[0], value[2])

    # T -> F
    @_('F')
    def T(self, value):
        return value[0]

    # F-> NUMBER
    @_('NUMBER')
    def F(self, value):
        return Num(value[0])

    # F-> ID
    @_('ID')
    def F(self, value):
        return Var(value[0])

    # F -> (E)
    @_('"(" E ")"')
    def F(self, value):
        return value[1]


lexer = ExprAstLexer()
parser = ExprAstParser()
inp = 'x=a + b*c-a/b+c*4'
inp=inp.replace(' ','')
result = parser.parse(lexer.tokenize(inp))
#print(pretty(result))
#to_dot(result)
tripleprogram = generate_for_statement(result)
tac = render_threeAddressCode(tripleprogram)
print(tac)
mipscode = generate_mips(tripleprogram)
print(mipscode)

