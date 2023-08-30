import os
import numpy as np


def save_random_indices(validation_percent, test_percent, dataset_filepath, indices_filepath):
    if validation_percent + test_percent > 1:
        print('Sum of percents greater 1')
        return

    validation_class_indices = []
    test_class_indices = []

    for class_name in [c_class.name for c_class in os.scandir(dataset_filepath) if c_class.is_dir()]:
        class_size = len([file.name for file in os.scandir(os.path.join(dataset_filepath, class_name))])
        validation_size = int(class_size * validation_percent)
        test_size = int(class_size * test_percent)
        indices = np.random.choice(np.arange(class_size), size=validation_size + test_size, replace=False)
        validation_class_indices.append(indices[:validation_size])
        test_class_indices.append(indices[validation_size:])

    indices = {'validation_class_indices': validation_class_indices, 'test_class_indices': test_class_indices}
    np.savez(file=indices_filepath, indices=indices)


if __name__ == '__main__':
    save_random_indices(0.1, 0.1, 'G:/Datasets/Planet/Texture/complete', 'random_indices_p.npz')
