import numpy as np
from goph547lab00.arrays import (
    square_ones,
    )

def main():
    # test creating square array of ones
    print('Part A')
    A_np = np.ones((3,3))
    A = square_ones(3)
    print(f'A_np:\n{A_np}')
    print(f'A:\n{A}')

    print('Part B')

    B_np = np.ones((3,5))
    print(f'1) B_np:\n{B_np}')

    C_np = np.full((6,3), np.nan)
    print(f'2) C_np:\n{C_np}')

    start = 43; end = 75

    D_np = np.arange(start, end, 2).reshape(-1, 1)  # -1 1 for column, 1 -1 for row
    print(f'3) D_np:\n{D_np}')

    D_sum = D_np.sum()
    print(f'4) Sum of D_np:\n{D_sum}')

    E_np = np.array([5, 7, 2, 1, -2, 3, 4, 4, 4]).reshape(3, 3)
    print(f'5) Custom array A:\n{E_np}')

    print(f'6) Custom array B:\n{np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)}')

    F_np = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)

    G_np = E_np * F_np
    print(f'7) Element-wise multiplication:\n{G_np}')

    dot_prod = E_np @ F_np
    print(f'8) Dot product:\n{dot_prod}')

    cross_prod = np.cross(E_np, F_np)
    print(f'9) Cross product:\n{cross_prod}')

if __name__ == '__main__':
    main()