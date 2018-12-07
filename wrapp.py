from time import time



def decorator1(func):
	def wrapper1():
		t1 = time()
		func()
		t2 = time()
		print('Running %s takes %s seconds' % (func.__name__, str(t2 - t1)))
	return wrapper1

@decorator1
def f1():
	print('test inside f1')



class decorator2:
	def __init__(self, func):
		self._func = func 
		print('flag1************')

	def __call__(self):
		print('flag2')
		t1 = time()
		self._func()
		t2 = time()
		print('Running %s cost %s seconds ' % (self._func.__name__, str(t2-t1)) )

@decorator2
def f2():
	print('test inside %s' % 'f2')




if __name__ == '__main__':
	f1()
	#f2()


