

class DotDict(dict):
    """ A dictionary that allows item = d.key access for brevity. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        for k, v in kwargs.items():
            self[k] = v


    def from_dict(self, d, recursive=False):
        for k, v in d.items():

            if recursive and isinstance(v, dict):
                self[k] = DotDict().from_dict(v, recursive=True)

            elif recursive and (isinstance(v, list) or (isinstance(v, np.ndarray) and isinstance(v[0], dict))):
                self[k] = []
                for it in v:
                    if isinstance(it, dict):
                        self[k].append(DotDict().from_dict(it, recursive=True))
                    else:
                        self[k].append(it)

            else:
                self[k] = v

        return self


    def to_dict(self, recursive=False):
        d = dict()
        for k, v in self.items():

            if recursive and isinstance(v, dict):
                d[k] = DotDict().to_dict(v, recursive=True)

            elif recursive and (isinstance(v, list) or (isinstance(v, np.ndarray) and isinstance(v[0], dict))):
                d[k] = []
                for it in v:
                    if isinstance(it, dict):
                        d[k].append(DotDict().to_dict(it, recursive=True))
                    else:
                        d[k].append(it)

            else:
                d[k] = v

        return d


    def __getattr__(self, k, *args, **kwargs):
        try:
            return super().__getattr__(k)
        except AttributeError:

            try:
                return self[k]
            except KeyError:

                if len(args) > 0:
                    return args[0]
                elif "default" in kwargs:
                    return kwargs["default"]
                else:
                    raise AttributeError


    def __setattr__(self, k, v):
        try:
            super().__getattr__(k)
            super().__setattr__(k, v)
        except AttributeError:
            self[k] = v
            