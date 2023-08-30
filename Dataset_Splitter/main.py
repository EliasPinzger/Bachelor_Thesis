import os
import numpy as np


class Splitter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.class_names = [c_class.name for c_class in os.scandir(self.filepath) if c_class.is_dir()]

    def __create_folder_structure__(self, validation_set_filepath, test_set_filepath):
        os.mkdir(validation_set_filepath)
        os.mkdir(test_set_filepath)

        for class_name in self.class_names:
            os.mkdir(os.path.join(validation_set_filepath, class_name))
            os.mkdir(os.path.join(test_set_filepath, class_name))

    def split(self, validation_class_indices, test_class_indices):
        validation_set_filepath = os.path.join(os.path.dirname(self.filepath), 'val')
        test_set_filepath = os.path.join(os.path.dirname(self.filepath), 'test')
        self.__create_folder_structure__(validation_set_filepath, test_set_filepath)

        for class_index, class_name in enumerate(self.class_names):
            source_class_filepath = os.path.join(self.filepath, class_name)
            validation_class_filepath = os.path.join(validation_set_filepath, class_name)
            test_class_filepath = os.path.join(test_set_filepath, class_name)
            file_names = [file.name for file in os.scandir(source_class_filepath)]

            for validation_index in validation_class_indices[class_index]:
                os.rename(os.path.join(source_class_filepath, file_names[validation_index]),
                          os.path.join(validation_class_filepath, file_names[validation_index]))

            for training_index in test_class_indices[class_index]:
                os.rename(os.path.join(source_class_filepath, file_names[training_index]),
                          os.path.join(test_class_filepath, file_names[training_index]))

        os.rename(self.filepath, os.path.join(os.path.dirname(self.filepath), 'train'))


if __name__ == '__main__':
    splitter = Splitter('G:/Datasets/Planet/LightDirection_No_Clouds/complete')
    indices = np.load('random_indices_p.npz', allow_pickle=True)['indices']
    splitter.split(indices.item().get('validation_class_indices'), indices.item().get('test_class_indices'))
    print('Dataset splitting finished')

