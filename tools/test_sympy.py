from sympy.parsing.latex import parse_latex
from sympy import symbols
from sympy import solve
from sympy import simplify, trigsimp
from sympy import evalf
from sympy import pi
from sympy import latex
from timeout import timeout


@timeout(seconds=20)
def compute(mode, latex_expr, *args):
    latex_expr = latex_expr.strip().strip('$')
    def wrap_math(r):
        return '$' + str(r) + '$'
    try:
        if mode == 'calculate':
            res = parse_latex(latex_expr).evalf(3, subs={
                'pi': pi
            })
            return wrap_math(latex(res))

        elif mode == 'simplify':
            res = simplify(parse_latex(latex_expr))
            return wrap_math(latex(res))

        elif mode.startswith('solve '):
            var = mode.split('solve ')[-1].strip()
            res = solve(parse_latex(latex_expr), symbols(var))
            return wrap_math(latex(res))

        else:
            return 'Error: Wrong input format, try again!'

    except Exception as err:
        return 'Error: ' + str(err)


if __name__ == '__main__':
    res = compute('solve x', r'\sin x = 1/2')
    print(res)
    #res = compute('simplify', r'x + y - 3 + 2 + x')
    #print(res)
    #res = compute('calculate', r'\pi + \sin(\pi / 2)')
    #print(res)
    #res = compute('foo', 'bar')
    #print(res)
    #res = compute("solve y", "$y = 1 - 2 y^2$")
    #print(res)
    #res = compute("solve x", "$\\sin x = -1$")
    #print(res)
    #res = compute("solve x", "$\\sin x = 1/2$")
    #print(res)
    #res = compute('simplify', r'$\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 8 \\ -1 \end{pmatrix} + t \begin{pmatrix} 2 \\ 3 \end{pmatrix}$')
    #print(res)

    #print(compute('solve y', '$y^9+4y^6-4y^3-8$', 'foo'))
