from time import time
import logging 
## https://www.zhihu.com/question/26930016

############	1. simple decorator  #########
def decorator1(func):
	print('flag0====')
	def wrapper1():
		t1 = time()
		func()
		t2 = time()
		print('Running %s takes %s seconds' % (func.__name__, str(t2 - t1)))
	return wrapper1

@decorator1
def f1():
	print('test inside f1')


############	2. "class" decorator  #########
class decorator2:
	def __init__(self, func):
		self._func = func 
		print('decorator2 definition************')

	def __call__(self):
		print('decorator2 calling')
		t1 = time()
		self._func()
		t2 = time()
		print('Running %s cost %s seconds ' % (self._func.__name__, str(t2-t1)) )

@decorator2
def f2():
	print('test inside %s' % 'f2')


############	3. decorator with (parameter in function)  #########
def decorator3(func):
	print('decorator3 definition=====')
	def wrapper1(*args, **kwargs):
		print('decorator3 calling====')
		return func(*args, **kwargs)
	return wrapper1

#### formulation I
@decorator3
def f3(batchsize, batch_first = True, datatype = 'str'):
	assert batch_first == True
	print(datatype)
	return batchsize


#### formulation II
def f3(batchsize, batch_first = True, datatype = 'str'):
	assert batch_first == True
	print(datatype)
	return batchsize
f3 = decorator3(f3)


###!!!! formulation I and formulation II are equivalent 

############	4. parameter in decorator  #########
### e.g., https://github.com/SwordYork/DCNMT/blob/old-version/model.py


def decorator4(level):
	print('decorator4---- 1. %s ' % level)
	def f2(func):
		print('decorator4---- 2. %s %s' % (level, func.__name__))
		def f3(*args, **kwargs):
			print('%s begin ' % func.__name__)
			print('decorator4---- 3. %s %s' % (level, func.__name__))
			a = func(level, *args, **kwargs)
			print('%s end' % func.__name__)
			return a 
		return f3
	return f2

### formulation I
@decorator4(level = 'INFO')
def f4(lev, a = 1, b = 2):
	##  print('inside function:::: %s' % level)  ### WRONG
	print(lev)
	return a + b 

### formulation II 
def f4(lev, a = 1, b = 2):
	##  print('inside function:::: %s' % level)  ### WRONG
	print(lev)
	return a + b 
### II.1
f4 = decorator4(level = 'INFO')(f4)
### II.2
tmp = decorator4(level = 'INFO')
f4 = tmp(f4)
###!!!! formulation I and formulation II are equivalent 


print('======================test f5======================')

@decorator3
@decorator4(level = 'INFO')
def f5(lev, a = 4, b = 3):
	print('f5 %s' % lev)
	return a ** b 



print('======================test f6======================')

## how to use wrapper 
## https://github.com/mila-udem/blocks/blob/master/blocks/bricks/base.py
## line 915 
## blocks/blocks/bricks/recurrent/base.py 
## https://github.com/mila-udem/blocks/blob/master/blocks/bricks/recurrent/base.py
## line-63:  def recurrent(*args, **kwargs): 


def application(*args, **kwargs):
	assert not args and kwargs
	kwa = kwargs
	def wrap_application(Application_class):
		def f3(*args, **kwargs):
			app = Application_class(*args, **kwargs)
			for key, value in kwa.items():
				setattr(app, key, value)
				print('{}  {}'.format(key, value))
			return app 
		return f3
	return wrap_application

@application(attr3 = 3)
class Application:
	def __init__(self, attr1, attr2 = 2):
		self._attr1 = attr1
		self._attr2 = attr2

	def print_info(self):
		print('attribute 1 is {}'.format(self._attr1))



###### 不带参数。装饰器。
def use_logging(func):
	def wrapper(*args, **kwargs):
		logging.warn("%s is running" % func.__name__)
		return func(*args)
	return wrapper

@use_logging
def foo():
	print("i am foo")






###### 带参数。装饰器。

def use_logging(level):
	def decorator(func):
		def wrapper(*args, **kwargs):
			if level == 'warning':
				logging.warn('%s is running' % func.__name__)
			return func(*args)
		return wrapper
	return decorator

#@use_logging(level = 'warning')
def foo(name = 'fool'):
	print('i am %s' % name)


foo = use_logging(level = 'warning')(foo)   ##### 等价修饰器 ！！好理解   括号从左到右， 函数调用从外到内 
##use_logging(level = 'warning')(foo)()  ### 或者 直接调用 


#### 带 . 运算的装饰器  


if __name__ == '__main__':
	#f1()
	#f2()
	#assert f3(batchsize = 16, batch_first = True) == 16
	#print('======================test f4======================')
	#assert f4(3,2) == 5
	#print('======================test f5======================')
	#print(f5(2,5))
	app = Application(1)
	assert hasattr(app, 'attr3')

	pass
	foo()
	



