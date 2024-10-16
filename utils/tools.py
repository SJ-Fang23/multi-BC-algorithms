

# every, once, until from dreamer-pytorch
class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until
    
def recursive_update_dict(d, u):
    '''
    Recursively update a dictionary with another dictionary.
    
    Args:
    - d (dict): The dictionary to update.
    - u (dict): The dictionary to update with.
    '''
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d
