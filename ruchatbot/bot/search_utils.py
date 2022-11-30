def normalize_for_lookup(s):
    us = s.lower().replace('ё', 'е')
    if us[-1] in '.!?;':
        us = us[:-1]
    return us


def search_among(needle_str, hay):
    uneedle = normalize_for_lookup(needle_str)
    return any((uneedle == normalize_for_lookup(s)) for s in hay)
