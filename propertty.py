

class Exam(object):
	def __init__(self, val):
		self._score = val 

	@property
	def score1(self):
		return self._score  ### read-only 

	
	@score1.setter
	def score1(self, val):
		self._score = val 
	
	@score1.deleter
	def score1(self, val):
		del self._score

	@staticmethod
	def func2(a,b):
		return a + b
	## static-method,  
	## note that there is no self
	##  http://www.runoob.com/python/python-func-staticmethod.html

	@classmethod
	def method_of_class(clss, val):
		print(clss)
	## https://zhuanlan.zhihu.com/p/28010894 



if __name__ == '__main__':
	a = Exam(3)
	print(a.score1)
	a._score = 4
	#del a.score1
	print(a.score1)
	print(Exam.func2(3,4))
	print(a.func2(5,4))
	a.method_of_class(1)
	Exam.method_of_class(2)

###  http://wiki.jikexueyuan.com/project/explore-python/Class/property.html
## http://www.runoob.com/python/python-func-property.html
##

'''
most important is 

	1. @property def xxx(self): => xxx is read-only attribute
	2. @property + xxx.setter => readable + modifiable 
	3. 


'''

