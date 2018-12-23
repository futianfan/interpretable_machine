
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

def func4(*argss, **kwargss):
	print(locals())
	
def func5(a,b,c,d):
	print(locals())
	value = list(v for k,v in locals().items())
	return sum(value)


if __name__ == '__main__':
	##print(func1(1,2,3,4))
	##a = {str(i):i**2 for i in range(4)}
	##b = func2(**a)
	
	### func3
	'''
	dict_a = {'trg_size': 10, 'src_size': 20 }
	b = func3(**dict_a)
	print(b)
	'''

	### func4
	'''
	lst = [1,2,3]
	#  kwargs = {i:i+10 for i in range(3)}		## this is wrong, kwargs' key must be string!!
	kwargs = {str(i):i+10 for i in range(3)}
	func4(1, 2, *lst, **kwargs)
	'''


	### func5.1
	keyy = 'abcd'
	valuee = [1,2,3,4,5]
	kwargss = {i:j for i,j in zip(keyy, valuee)}
	print(func5(**kwargss))

	### func5.2
	lst = [1,2,3,4]
	print(func5(*lst))


	### func5.3
	lst = [1,2]
	kwargss = {'c':5, 'd':6}
	assert func5(*lst, **kwargss) == 14




















