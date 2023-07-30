from sympy.parsing.latex import parse_latex
from sympy import symbols
from sympy import solve
from sympy import simplify, trigsimp
from sympy import evalf
from sympy import pi
from sympy import latex


def compute(mode, latex_expr):
    latex_expr = latex_expr.strip().strip('$')
    if mode == 'calculate':
        res = parse_latex(latex_expr).evalf(3, subs={
            'pi': pi
        })
        return str(latex(res))

    elif mode == 'simplify':
        res = simplify(parse_latex(latex_expr))
        return str(latex(res))

    elif mode.startswith('solve'):
        var = mode.split('solve ')[-1].strip()
        res = solve(parse_latex(latex_expr), symbols(var))
        return str(latex(res))

    else:
        return 'Wrong input format, try again!'


if __name__ == '__main__':
    res = compute('solve x', r'\sin x = 1/2')
    print(res)
    res = compute('simplify', r'x + y - 3 + 2 + x')
    print(res)
    res = compute('calculate', r'\pi + \sin(\pi / 2)')
    print(res)
    res = compute('foo', 'bar')
    print(res)
    res = compute("solve y", "$y = 1 - 2 y^2$")
    print(res)
    res = compute("solve x", "$\\sin x = -1$")
    print(res)
    res = compute("solve x", "$\\sin x = 1/2$")
    print(res)
