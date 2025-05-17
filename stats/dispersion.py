def range_(data):
    return max(data) - min(data)

def variance(data):
    m = sum(data) / len(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)

def std_deviation(data):
    import math
    return math.sqrt(variance(data))

def iqr(data):
    from statistics import median
    sorted_data = sorted(data)
    mid = len(data) // 2
    lower = sorted_data[:mid]
    upper = sorted_data[mid:] if len(data) % 2 == 0 else sorted_data[mid+1:]
    return median(upper) - median(lower)




def variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("Variance requires at least two data points.")
    
    mean_value = sum(data) / n
    squared_diffs = [(x - mean_value) ** 2 for x in data]
    return sum(squared_diffs) / (n - 1)  # sample variance






def detect_outliers_iqr(data):
    sorted_data = sorted(data)
    n = len(data)
    mid = n // 2

    # Split data into lower and upper halves
    lower_half = sorted_data[:mid]
    upper_half = sorted_data[mid:] if n % 2 == 0 else sorted_data[mid+1:]

    def median(d):
        m = len(d)
        half = m // 2
        return (d[half - 1] + d[half]) / 2 if m % 2 == 0 else d[half]

    Q1 = median(lower_half)
    Q3 = median(upper_half)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers

