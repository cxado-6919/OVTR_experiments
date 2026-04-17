class DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only

    def __repr__(self):
        return f"DataContainer({self.data!r})"
