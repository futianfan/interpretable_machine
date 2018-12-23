a = [[1,2,3], [4,5,6,7]]

print(list(zip([1])))

b, c, d = zip(*a)
assert b == (1,4) and c == (2,5)

print(list(zip(*a)))

print(list(zip(a)))


b = [[1,2,3], iter([4,5,6,7]), [8,9]]
print(list(zip(*b)))


