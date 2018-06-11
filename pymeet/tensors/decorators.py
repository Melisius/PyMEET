class memorize:
    def __init__(self, function):
        self.function = function
        self.memorized = {}

    def __call__(self, *args):
        try:
            return self.memorized[args]
        except KeyError:
            self.memorized[args] = self.function(*args)
            return self.memorized[args]


class memo1:
    def __init__(self, function):
        self.function = function
        self.memorized = {}

    def __call__(self, *args):
        try:
            return self.memorized[args[1]]
        except KeyError:
            self.memorized[args[1]] = self.function(*args)
            return self.memorized[args[1]]
