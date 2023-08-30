import numpy as np

if __name__ == '__main__':
    np.savez(
        file='../resources_p/random_attributes_values_no_clouds.npz',
        heights=np.sort(np.random.choice(np.linspace(0.5, 0.5, 1, endpoint=True), size=1, replace=False)),
        betas=np.sort(np.random.choice(np.linspace(0, 0, 1, endpoint=True), size=1, replace=False)),
        gammas=np.sort(np.random.choice(np.linspace(0, 0, 1, endpoint=True), size=1, replace=False)),
        scales=np.sort(np.random.choice(np.linspace(1, 1.5, 48, endpoint=True), size=24, replace=False)),
        light_energies=np.sort(np.random.choice(np.linspace(1, 3, 32, endpoint=True), size=16, replace=False)),
        light_directions=np.sort(np.random.choice(np.linspace(-100, 80, 6, endpoint=True), size=6, replace=False)))
