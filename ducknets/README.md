The pickled ducknet was made as follows:

>>> import ducknet
>>> dn = ducknet.DuckNetwork([1,2,3],["out"],[])
>>> import pickle
>>> with open('ducknet.pkl', 'wb') as output:
...     pickle.dump(dn, output, pickle.HIGHEST_PROTOCOL)
...


quack quack.
