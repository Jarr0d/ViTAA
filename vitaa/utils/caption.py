import torch


class Caption(object):

    def __init__(self, text, length=None, max_length=None, padded=False, dtype=torch.int64):
        device = text.device if isinstance(text, torch.Tensor) else torch.device("cpu")
        if isinstance(text, list):
            text = [torch.as_tensor(line, dtype=dtype, device=device)
                    for line in text]
            if length is None:
                length = torch.stack([torch.tensor(line.size(0), dtype=torch.int64, device=device)
                                      for line in text])
            if max_length is None:
                max_length = max([line.size(-1) for line in text])
        else:
            text = torch.as_tensor(text, dtype=dtype, device=device)
            if length is None:
                length = torch.tensor(text.size(-1), dtype=torch.int64, device=device)
            if max_length is None:
                max_length = text.size(-1)

        if not padded:
            text = self.pad(text, max_length, device)

        self.text = text
        self.length = length
        self.max_length = max_length
        self.padded = True
        self.dtype = dtype
        self.extra_fields = {}

    def pad(self, text, max_length, device):
        padded = []
        for line in text:
            length = line.size(0)
            if length < max_length:
                pad = torch.zeros((max_length - length), dtype=torch.int64, device=device)
                padded.append(torch.cat((line, pad)))
        return torch.stack(padded)

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    # Tensor-like methods

    def to(self, device):
        cap = Caption(self.text.to(device), self.length.to(device), self.max_length, self.padded, self.dtype)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            cap.add_field(k, v)
        return cap

    def __getitem__(self, item):
        cap = Caption(self.text[item], self.max_length, self.padded)
        for k, v in self.extra_fields.items():
            cap.add_field(k, v[item])
        return cap

    def __len__(self):
        return self.text.shape[0]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "length={}, ".format(self.length)
        s += "max_length={}, ".format(self.max_length)
        s += "padded={}, ".format(self.padded)
        return s