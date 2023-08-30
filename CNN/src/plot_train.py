import os
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import rc
from tensorflow.python.summary.summary_iterator import summary_iterator

rc('text', usetex=True)
rc('font', family='Latin Modern Roman', size=11)


def sort(data, order):
    result = []
    for o in order:
        for d in data:
            if o[0] == d[1]:
                result.append((d[0], o[1], o[2]))
    return result


def plot(filepath, tag, order=None):
    data = []
    for path, _, files in os.walk(filepath):
        for name in files:
            if 'events.out.tfevents' in name:
                event_values = []
                for event in summary_iterator(os.path.join(path, name)):
                    for value in event.summary.value:
                        if value.tag == tag:
                            event_values.append(value.simple_value)
                data.append((event_values, str(Path(path).name)))

    if order is not None:
        data = sort(data, order)

    for d in data:
        plt.plot(d[0], label=d[1], marker='o', color=d[2])

    plt.xticks(range(0, 31, 5))
    plt.yticks(range(0, 101, 20))
    plt.margins(x=0.06)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(tag.replace('/', '_').lower() + '.pdf', dpi=300, transparent=True, facecolor=(1, 1, 1),
                edgecolor='none')


if __name__ == '__main__':

    # order: [('name in events file', 'name in plot', 'color')]

    order_g = [('Shape', 'GS-shape', '#1f77b4'),
               ('Texture', 'GS-texture', '#ff7f0e'),
               ('Color', 'GS-color', '#2ca02c'),
               ('Shape_Texture_Color', 'GS-all', '#d62728'),
               ('Shape_Texture', 'GS-shape-texture', '#9467bd')]

    order_p = [('LightDirection', 'P-lightDirection', '#1f77b4'),
               ('Texture', 'P-texture', '#ff7f0e'),
               ('Color', 'P-color', '#2ca02c'),
               ('LightDirection_Texture_Color', 'P-all', '#d62728'),
               ('LightDirection_Texture', ' P-lightDirection-texture', '#9467bd')]

    order_p_nc = [('LightDirection', 'P-lightDirection', '#1f77b4'),
                  ('Texture', 'P-texture', '#ff7f0e'),
                  ('LightDirection_Texture', ' P-lightDirection-texture', '#9467bd'),
                  ('LightDirection_NoClouds', 'P-lightDirection-noClouds', '#8c564b'),
                  ('Texture_NoClouds', 'P-texture-noClouds', '#e377c2'),
                  ('LightDirection_Texture_NoClouds', ' P-lightDirection-texture-noClouds', '#7f7f7f')]

    order_g_s = [('Shape_Texture', 'GS-shape-texture', '#9467bd'),
                 ('Shape_Texture_0.01', 'GS-shape-texture-0.01', '#8c564b'),
                 ('Shape_Texture_0.1',  'GS-shape-texture-0.1', '#e377c2'),
                 ('Shape_Texture_0.15',  'GS-shape-texture-0.15', '#7f7f7f'),
                 ('Shape_Texture_0.2',  'GS-shape-texture-0.2', '#bcbd22'),
                 ('Shape_Texture_0.25',  'GS-shape-texture-0.25', '#17becf'),
                 ('Shape_Texture_0.3',  'GS-shape-texture-0.3', '#abcdef')]

    order_g_t = [('Shape_90', 'GS-shape-90', '#8c564b'),
                 ('Texture_90', 'GS-texture-90', '#e377c2'),
                 ('Transfer_Learning_Shape', 'GS-shape-transferLearning', '#7f7f7f'),
                 ('Transfer_Learning_Texture', 'GS-texture-transferLearning', '#bcbd22')]

    plot('../Models/Transfer_Learning', 'Accuracy/val', order_g_t)
