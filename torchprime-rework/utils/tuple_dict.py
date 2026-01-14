

class TupleDict:
    """
    Kind of like a dictionary, but only using tuples.

    Only supports adding new items, not modifying existing ones.
    """

    def __init__(self):
        self._keys = []
        self._values = []

    

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Keys must be strings.")
        
        if key in self._keys:
            raise KeyError(f"Key '{key}' already exists. TupleDict does not support modifying existing keys.")
        
        self._keys.append(key)
        self._values.append(value)

    
    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Keys must be strings.")
        
        if key not in self._keys:
            raise KeyError(f"Key '{key}' does not exist.")
        
        index = self._keys.index(key)
        return self._values[index]
    

    def keys(self):
        return tuple(self._keys)
    
    def values(self):
        return tuple(self._values)
    