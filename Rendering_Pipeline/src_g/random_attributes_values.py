import numpy as np

if __name__ == '__main__':
    np.savez(
        file='../resources_g/random_attributes_values.npz',
        heights=np.sort(np.random.choice(np.linspace(14, 14, 1, endpoint=True), size=1, replace=False)),
        betas=np.sort(np.random.choice(np.linspace(50, -50, 20, endpoint=True), size=4, replace=False)),
        gammas=np.sort(np.random.choice(np.linspace(0, 360, 20, endpoint=False), size=4, replace=False)),
        scales=np.sort(np.random.choice(np.linspace(1, 1.5, 20, endpoint=True), size=5, replace=False)),
        light_energies=np.sort(np.random.choice(np.linspace(10000, 35000, 20, endpoint=True), size=5, replace=False)))
