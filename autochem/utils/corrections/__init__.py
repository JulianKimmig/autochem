def norm_01(x):
    unf1 = x.min()
    x = x - x.min()
    unf2 = x.max()
    x = x / x.max()

    def _unnorm(x):
        x = x * unf2
        x = x + unf1
        return x

    return x, {"unnorm": _unnorm}


def norm_data(x):
    return norm_01(x)
