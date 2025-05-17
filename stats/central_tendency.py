def mean(data):
    return sum(data) / len(data)

def median(data):
    sorted_data = sorted(data)
    n = len(data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

def mode(data):
    from collections import Counter
    freq = Counter(data)
    max_count = max(freq.values())
    return [k for k, v in freq.items() if v == max_count]
