import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

    img = np.asarray(Image.open('../examples/rock_canyon.jpg'))
    print(f'Regular image: {repr(img)}')  # shape is 296, 474, 3
    imgplot = plt.imshow(img)  # creates plot
    plt.show()  # displays plot

    grey_img = np.asarray(Image.open('../examples/rock_canyon.jpg').convert('L'))
    print(f'Greyscale image: {repr(grey_img)}')  # shape is 296, 474. no 3rd entry bc not RGB
    grey_imgplot = plt.imshow(grey_img)
    plt.show()

    small_grey_img = grey_img[150:240, 110:150]
    small_grey_imgplot = plt.imshow(small_grey_img)
    plt.show()

    R_x = img[:, :, 0].mean(axis=0); G_x = img[:, :, 1].mean(axis=0); B_x = img[:, :, 2].mean(axis=0)
    RGB_x = img.mean(axis=(0, 2))

    R_y = img[:, :, 0].mean(axis=1); G_y = img[:, :, 1].mean(axis=1); B_y = img[:, :, 2].mean(axis=1)
    RGB_y = img.mean(axis=(1, 2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(img.shape[1])
    ax[0].plot(x, R_x, 'r', label='R'); ax[0].plot(x, G_x, 'g', label='G'); ax[0].plot(x, B_x, 'b', label='B')
    ax[0].plot(x, RGB_x, 'k', linewidth=4, label='Mean RGB')

    ax[0].set_xlabel('x-coord'); ax[0].set_ylabel('colour value')
    ax[0].legend()
    ax[0].set_title('Colour values as a function of x-coordinates')

    y = np.arange(img.shape[0])
    ax[1].plot(R_y, y, 'r', label='R'); ax[1].plot(G_y, y, 'g', label='G'); ax[1].plot(B_y, y, 'b', label='B')
    ax[1].plot(RGB_y, y, 'k', linewidth=4, label='Mean RGB')

    ax[1].set_xlabel('colour value'); ax[1].set_ylabel('y-coord')
    ax[1].legend()
    ax[1].set_title('Y-coordinates as a function of Colour Values')

    plt.tight_layout()
    plt.savefig('../examples/rock_canyon_RGB_summary.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()