def pearson_corr(x, y):
    if len(x) != len(y):
        raise ValueError("Datasets must be the same length.")
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = (
        sum((xi - mean_x) ** 2 for xi in x) *
        sum((yi - mean_y) ** 2 for yi in y)
    ) ** 0.5
    return numerator / denominator
