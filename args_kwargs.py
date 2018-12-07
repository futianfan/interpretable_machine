
def func1(a, *b):
	summ = 0
	summ += a
	for i in b:
		summ += i
	return summ

def func2(**ddic):
	ddic['s'] = -1 
	print(ddic)

def func3(trg_size, src_size):
	return trg_size + src_size

if __name__ == '__main__':
	##print(func1(1,2,3,4))
	##a = {str(i):i**2 for i in range(4)}
	##b = func2(**a)
	
	dict_a = {'trg_size': 10, 'src_size': 20 }
	b = func3(**dict_a)
	print(b)

