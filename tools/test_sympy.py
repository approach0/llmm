from sympy.parsing.latex import parse_latex
from sympy import symbols
from sympy import solve


if __name__ == '__main__':
    res = solve(parse_latex(r'x=1-2x^2'), symbols('x'))
    print(res)
