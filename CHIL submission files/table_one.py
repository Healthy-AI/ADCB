def _col_str(d, t):
    if t == 'cont' or t == 'intcat':
        return '%.1f (%.1f)' % (d.mean(), d.std())
    elif t == 'bin' or t == 'cat':
        if d.shape[0] > 0:
            return '%d (%.1f%%)' % (d.sum(), d.mean() * 100)
        else:
            return '- (-)'


def table_one(Ds, spec, labels):
    out = '\\toprule ' + ' & ' + \
        ' & '.join(['%s, n=%d' % (labels[i], Ds[i].shape[0])
                   for i in range(len(Ds))]) + '\\\\ \n'

    for k, cs in spec.items():
        out += '\midrule \n' + k + '\\\\ \n\midrule \n'
        for s in cs:

            out += s['label']
            if s['type'] == 'cat':
                out += ' \\\\'

            cols = []
            if s['type'] == 'cont':
                out += ' & ' + \
                    ' & '.join([_col_str(D[s['key']].dropna(), 'cont')
                               for D in Ds])
            elif s['type'] == 'bin':
                out += ' & ' + \
                    ' & '.join([_col_str(D[s['key']].dropna() == 1, 'bin')
                               for D in Ds])
            elif s['type'] == 'cat':
                cat = s['cats']
                out += '\n'
                for i in range(len(cat)):
                    out += '\ \ %s & ' % cat[i] + ' & '.join(
                        [_col_str(D[s['key']].dropna() == i, 'cat') for D in Ds])
                    out += ' \\\\ \n'
            elif s['type'] == 'intcat':
                cat = s['cats']
                out += ' & ' + ' & '.join([_col_str(D[s['key']].dropna()
                                                    .replace(dict([(i, cat[i]) for i in range(len(cat))])), 'intcat') for D in Ds])
            if not s['type'] == 'cat':
                out += ' \\\\ \n'
    return out
