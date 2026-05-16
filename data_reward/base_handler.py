
import datasets

class BaseHandler:

    url: str = None
    subset: str | list = None
    split: str | list = None

    domain: str = None
    reward_type: str = None

    verification_mode = None


    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.post_init()

    def post_init(self):
        pass
    

    def load_dataset(self):
        if self.subset is not None and self.split is not None:
            assert not (isinstance(self.subset, list) and isinstance(self.split, list)), "Cannot have both subset and split as lists."

        if self.subset is not None and isinstance(self.subset, list):

            subs = [
                datasets.load_dataset(self.url, sub, split=self.split, verification_mode=self.verification_mode)
                for sub in self.subset
            ]
            
            common_columns = set.intersection(*[set(ds.column_names) for ds in subs])
            subs = [
                ds.remove_columns(
                    [col for col in ds.column_names if col not in common_columns]
                )
                for ds in subs
            ]

            return datasets.concatenate_datasets(subs)

        if self.split is not None and isinstance(self.split, list):

            splits = [
                datasets.load_dataset(self.url, self.subset, split=s, verification_mode=self.verification_mode)
                for s in self.split
            ]

            common_columns = set.intersection(*[set(ds.column_names) for ds in splits])
            splits = [
                ds.remove_columns(
                    [col for col in ds.column_names if col not in common_columns]
                )
                for ds in splits
            ]

            return datasets.concatenate_datasets(splits)

        return datasets.load_dataset(self.url, self.subset, split=self.split, verification_mode=self.verification_mode)


    def name(self):
        if self.subset is not None and isinstance(self.subset, str):
            return f"{self.url}/{self.subset}"        

        if self.split is not None and isinstance(self.split, str) and self.split != "train":
            return f"{self.url}/{self.split}"

        return self.url


    def full_map(self, example):

        out = {
            "source": self.name(),
            "domain": self.domain,
            "reward_type": self.reward_type,
            "cluster_key": None,
            "input": None,
            "output": None,
            "reward": None,
        }

        m = self.map(example)
        if m is None:
            return out
        
        for k in m.keys():
            assert k in out, f"Invalid key '{k}' returned by map. Valid keys are: {list(out.keys())}"
            out[k] = m[k]

        if out["cluster_key"] is None:
            out["cluster_key"] = out["input"]

        return out


    def map(self, example) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")
    

    def filter(self, example):
        return (
            example["cluster_key"] is not None and
            example["input"] is not None and
            example["output"] is not None and
            example["reward"] is not None
        )
    