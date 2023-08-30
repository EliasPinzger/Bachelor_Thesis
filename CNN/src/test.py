import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='Latin Modern Roman', size=11)


class Tester:
    def __init__(self, filepath_model, filepath_data_set):
        self.image_name = str(Path(filepath_model).parent.name) + '___' + str(Path(filepath_data_set).name + '.pdf')

        self.batch_size = 64
        self.workers = 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = models.efficientnet_b0(weights=None, num_classes=6).to(self.device)
        self.model.load_state_dict(torch.load(filepath_model, map_location=self.device), strict=True)
        self.model.eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset = datasets.ImageFolder(
            os.path.join(filepath_data_set, 'test'),
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

        self.classes = dataset.classes
        self.predictions = []
        self.targets = []
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)

    def __plot_confusion_matrix__(self):
        accuracy = np.sum(np.asarray(self.targets) == np.asarray(self.predictions)) / len(self.targets) * 100
        matrix = confusion_matrix(self.targets, self.predictions, normalize='true')
        matrix = np.matrix.round(matrix, 3)
        heatmap = sns.heatmap(matrix, annot=True, cmap='cividis')
        heatmap.set_ylabel('True label')
        heatmap.set_xlabel('Predicted label\n\nAccuracy: {:.3g}\%'.format(accuracy))
        plt.gca().set_aspect('equal')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.savefig(self.image_name, dpi=300, transparent=True)

    def start(self):
        with torch.no_grad():
            for images, target in self.loader:
                images = images.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)

                self.targets += target.cpu()
                self.predictions += torch.max(self.model(images).data, 1)[1].cpu()
            self.__plot_confusion_matrix__()


if __name__ == '__main__':
    tester = Tester('../Models/Planet_NoClouds/LightDirection_Texture_NoClouds/model.pth.tar',
                    'G:/Datasets/Planet_NoClouds/Texture_NoClouds')
    tester.start()
