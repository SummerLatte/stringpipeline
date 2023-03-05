code = """
a = a + b
"""

a = 1
b = 2

my_locals = {"a": a, "b": b}

exec(code, globals(), my_locals)

a = my_locals["a"]

print(a)
