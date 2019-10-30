key = [[1, 0, 0, 0, 0, 1, 0, 1],
       [1, 1, 0, 1, 0, 1, 0, 0],
       [1, 0, 1, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1],
       [1, 1, 0, 0, 0, 1, 0, 0],
       [1, 0, 0, 1, 0, 0, 1, 1],
       [1, 1, 0, 0, 0, 0, 1, 0]]

"""
and -- multiplication
or -- summation
      [x, u, p, k, s]
      [y, v, q, l, z]
[a, b][(a & x) | (b & y), (a & u) | (b & v), (a & p) | (b & q), (a & k) | (b & l), (a & s) | (b & z)]
[c, d][(c & x) | (d & y), (c & u) | (d & v), (c & p) | (d & q), (c & k) | (d & l), (c & s) | (d & z)]
"""

msg = "Eagle has landed"
msg = list(map(lambda b: bin(b)[2:].split(), bytearray(msg, 'utf-8')))
print(msg)