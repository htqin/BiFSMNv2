import random
import torch


class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""
    def __init__(self, prop=0.5, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range
        self.prop = prop

    def __call__(self, data: torch.Tensor):
        if random.uniform(0, 1) <= self.prop:
            data = data * random.uniform(*self.amplitude_range)
        return data


class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""
    def __init__(self, time=1, sample_rate=16000):
        self.target_len = time * sample_rate

    def __call__(self, data: torch.Tensor):
        cur_len = data.shape[1]
        if self.target_len <= cur_len:
            data = data[:, :self.target_len]
        else:
            data = torch.nn.functional.pad(data, (0, self.target_len - cur_len))
        return data


class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""
    def __init__(self, prop=0.5, max_scale=0.2, sample_rate=16000):
        self.max_scale = max_scale
        self.sample_rate = sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            scale = random.uniform(-self.max_scale, self.max_scale)
            speed_fac = 1.0 / (1 + scale)
            data = torch.nn.functional.interpolate(data.unsqueeze(1),
                                                   scale_factor=speed_fac,
                                                   mode='nearest').squeeze(1)
        return data


# class StretchAudio(object):
#     """Stretches an audio randomly."""
#
#     def __init__(self, max_scale=0.2):
#         self.max_scale = max_scale
#
#     def __call__(self, data):
#         scale = random.uniform(-self.max_scale, self.max_scale)
#         data['samples'] = librosa.effects.time_stretch(data['samples'], 1+scale)
#         return data


class TimeshiftAudio(object):
    """Shifts an audio randomly."""
    def __init__(self, prop=0.5, max_shift_seconds=0.2, sample_rate=16000):
        self.shift_len = max_shift_seconds * sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            shift = random.randint(-self.shift_len, self.shift_len)
            a = -min(0, shift)
            b = max(0, shift)
            data = torch.nn.functional.pad(data, (a, b), "constant")
            data = data[:, :data.shape[1] - a] if a else data[:, b:]
        return data
