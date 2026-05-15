
import datasets

class BaseHandler:

    url: str = None
    subset: str | list = None
    split: str | list = None

    kind: str = None

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
        m = self.map(example)
        if len(m) == 3:
            inp, out, reward = m
            keep = True
        elif len(m) == 4:
            inp, out, reward, keep = m

        if inp is None or out is None:
            keep = False

        return {
            "source": self.name(),
            "kind": self.kind,
            "input": inp,
            "output": out,
            "reward": reward,
            "keep": keep
        }


    def map(self, example):
        raise NotImplementedError("Subclasses must implement this method.")
    

    def filter(self, example):
        return example["keep"]
    