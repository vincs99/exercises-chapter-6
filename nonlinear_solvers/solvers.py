"""A module providing numerical solvers for nonlinear equations."""


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using Newton-Raphson iteration.

    Solve f==0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the iteration.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using Newton iteration.
    """
    iter = 0
    while iter < max_its:
        try:
            new_x = x_0 - f(x_0)/df(x_0)
        except ZeroDivisionError:
            print('derivative zero!')
            break
        else:
            if abs(f(new_x)) < eps:
                return new_x
            else:
                iter = iter + 1
                x_0 = new_x

    raise ConvergenceError(f"Need more iters than {max_its}!")


def bisection(f, x_0, x_1, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using bisection.

    Solve f==0 using bisection starting with the interval [x_0, x_1]. f(x_0)
    and f(x_1) must differ in sign.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    x_0 : float
        The left end of the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using bisection.
    """
    if f(x_0)*f(x_1) > 0:
        raise ValueError(f"The values f({x_0}) and f({x_1}) have"
                         + "the same sign!")
    else:
        iter = 0
        while iter < max_its:
            mid_x = (x_0 + x_1)/2
            if abs(f(mid_x)) < eps:
                return mid_x
            else:
                iter = iter + 1
                if f(mid_x)*f(x_0) >= 0:
                    x_0 = mid_x
                else:
                    x_1 = mid_x
        raise ConvergenceError(f"Need more iters than {max_its}!")


def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the bisection
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """
    try:
        return newton_raphson(f, df, x_0, eps, max_its_n)
    except ConvergenceError:
        return bisection(f, x_0, x_1, eps, max_its_b)
