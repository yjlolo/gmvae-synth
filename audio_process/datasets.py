import os
import numpy as np
from torch.utils.data import Dataset

OMAX = 7
OMIN = 0


class ReadSOL(Dataset):
    def __init__(self, path_to_dataset, transform=None):
        target_ins = ['/'.join([path_to_dataset, f]) for f in os.listdir(path_to_dataset)
                      if f in ['Brass', 'Keyboards_fix', 'Strings', 'Winds']]

        wav_path = []
        for f in target_ins:
            for i in os.listdir(f):
                if not i.startswith('.'):
                    ins_path = '/'.join([f, i, 'ordinario'])
                    ins_path = ['/'.join([ins_path, j]) for j in os.listdir(ins_path) if ('.wav' in j) or ('.aif' in j)]
                    wav_path.append(ins_path)

        concat_wav_path = [i for sub_wav_path in wav_path for i in sub_wav_path]

        self.path_to_dataset = path_to_dataset
        self.wav_path = wav_path
        self.path_to_data = concat_wav_path
        self.transform = transform

    def _move_octave(self, idx):
        x = self.path_to_data[idx]
        fwav = x.split('/')[-1]
        pitch = fwav.split('-')[2]
        octave = int(pitch[-1])  # old octave
        up_rp = np.arange(octave + 1, OMAX + 1)
        down_rp = np.arange(OMIN, octave)
        rp = np.hstack([down_rp, up_rp])
        np.random.seed(111)
        y = np.random.choice(rp, 1, replace=False)  # new octave
        new_pitch = pitch[:-1] + str(y[0])
        shift_step = (y - octave) * 12
        return pitch, new_pitch, shift_step

    def __len__(self):
        return len(self.path_to_data)

    def __getitem__(self, idx):
        x = self.path_to_data[idx]
        x_pitch, x_augpitch, x_augstep = self._move_octave(idx)
        if self.transform:
            x = self.transform(x)

        # return x, idx, self.path_to_data[idx], x_pitch, x_augpitch, x_augstep
        return x, idx, self.path_to_data[idx]


# ins_map = {
#     'EH_nA': 0,
#     'Hn': 1,
#     'TpC': 2,
#     'Pno': 3,
#     'Vn': 4,
#     'Vc': 5,
#     'ASax': 6,
#     'Bn': 7,
#     'ClBb': 8,
#     'Fl': 9,
#     'Ob': 10
# }

# new map includes T-T and aif files
ins_map = {
    # English-Horn
    'EH_nA': 0,
    # French-Horn
    'Hn': 1,
    'Corf': 1,
    'Corm': 1,
    # Tenor-Trombone
    'trbt': 2,
    # Trumpet-C
    'TpC': 3,
    'trof': 3,
    'trom': 3,
    'trop': 3,
    # Piano
    'Pno': 4,
    # Violin
    'Vn': 5,
    # Cello
    'Vc': 6,
    # Alto-Sax
    'ASax': 7,
    # Bassoon
    'Bn': 8,
    'fagf': 8,
    'fagm': 8,
    'fagp': 8,
    # Clarinet
    'ClBb': 9,
    'clbb': 9,
    # Flute
    'Fl': 10,
    # Oboe
    'Ob': 11
}

pitchclass_map = {
    'A': 0,
    'A#': 1,
    'B': 2,
    'C': 3,
    'C#': 4,
    'D': 5,
    'D#': 6,
    'E': 7,
    'F': 8,
    'F#': 9,
    'G': 10,
    'G#': 11
}

dynamic_map = {
    'f': 0,
    'ff': 1,
    'mf': 2,
    'p': 3,
    'pp': 4
}


def extract_pitchclass(p):
    if len(p) == 2:
        return p[0]
    else:
        return p[0:2]


class CollectSOLSpec(Dataset):
    def __init__(self, path_to_dataset, transform=None):
        #path_to_data = sorted(['/'.join([path_to_dataset, f]) for f in os.listdir(path_to_dataset)
        #                       if '.pth' in f], key = lambda x: )
        path_to_data = sorted([f for f in os.listdir(path_to_dataset) if '.pth' in f],
                              key=lambda x: '-'.join(x.split('-')[:4]))
        pitch_set = sorted(set([f.split('-')[2] for f in path_to_data]))
        pitch_map = dict((i, j) for (i, j) in enumerate(pitch_set))

        self.pitch_map = {v: k for k, v in pitch_map.items()}
        self.ins_map = ins_map
        self.pitchclass_map = pitchclass_map
        self.dynamic_map = dynamic_map

        dict_label = {}
        dict_label['ins'] = [ins_map[f.split('-')[0]] for f in path_to_data]
        dict_label['pitch_class'] = [pitchclass_map[extract_pitchclass(f.split('-')[2])] for f in path_to_data]
        dict_label['pitch'] = [self.pitch_map[f.split('-')[2]] for f in path_to_data]
        dict_label['dyn'] = [dynamic_map[f.split('-')[3]] for f in path_to_data]

        path_to_data = ['/'.join([path_to_dataset, f]) for f in path_to_data]
        self.path_to_dataset = path_to_dataset
        self.path_to_data = path_to_data
        self.label = dict_label
        self.transform = transform

    def __len__(self):
        return len(self.path_to_data)

    def __getitem__(self, idx):
        f = self.path_to_data[idx]
        if self.transform:
            x = self.transform(f)
        else:
            x = f
        y_ins = self.label['ins'][idx]
        y_pitch_class = self.label['pitch_class'][idx]
        y_pitch = self.label['pitch'][idx]
        y_dyn = self.label['dyn'][idx]
        return x, [y_ins, y_pitch_class, y_pitch, y_dyn], idx


if __name__ == "__main__":
    R = ReadSOL('/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/')
    # C = CollectSOLSpec('/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/melspec-128')
    # C = CollectSOLSpec('/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/cqt')
