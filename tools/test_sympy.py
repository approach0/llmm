from sympy.parsing.latex import parse_latex
from sympy import symbols
from sympy import solve


if __name__ == '__main__':
    #res = solve(parse_latex(r'y = 1 - 2 y^2'), symbols('y'))
    #res = solve(parse_latex(r'\sin x = -1'), symbols('x'))
    res = solve(parse_latex(r'\sin x = 1/2'), symbols('x'))
    print(res)
