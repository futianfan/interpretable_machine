
4 is important, help interpret Python better. 


1. logg.py:   logging

2. propertty.py:   property  
	1. @property: read-only attribute
	2. @staticmethod: no "self";   static method in class, no "self"
	3. @classmethod: no "self"; self is replace by "class"

	in a class:
	@property
	def attr_1(self):
		return _attr_1

	@attr_1.setter
	def attr_1(self, value):
		return _attr_1 = value


3. args_kwargs.py:   *args && **kwargs
	args
	**kwargs  dictionary
		config file dict = 'src_size':10, 'trg_size':20
		**dict
		
4.  decorator/wrap:
	decorator with parameter
	@property / @attr1.setter / @staticmethod / 

5.  subprocess


6. from collections import Counter, defaultdict
	a = defaultdict(lambda: len(a))
	b = defaultdict(lambda: 0)
	counter = Counter(lst)	// dic = dict(counter)


7. sorted(dict)  lst.sort()
	sort() only to dictionary; sorted: for any iterable object.
	sorted(counter.items(), key = lambda x: (-x[1], x[0]))
	e.g., 
		a = [1,2,3,2,1,3,4,4,3,1,2,1,3,1,1,2,3]
		ca = Counter(a)
		sorted(ca.items(), key = lambda x:(-x[1], x[0]))
		sorted(ca.items(), key = lambda x:(-x[0], x[0]))



8. zip: zipp.py
	e.g., 
		dict(zip(words, counts))
		zip(words, range(len(words)))
		zip(*lst)

===========understand *some_list and **some_dict (key must be string)



#### TO DO ####
__future__   absolute_import, division, print_function, import unicode_literals

collections
	Counter, namedtuple, Sequence, OrderedDict, MutableMapping, defaultdict, deque


import re


__all__.    __dict__.    __call__.   __init__.py. 

dict: dict.values(), dict.keys(). dict.update().   dict.items(),  sorted(dict.items(), key = lambda x:(-x[1], x[0])). iteritems(). dict.get(2, defaultvalue), setdefault;     dict.pop(key_value)


iter().   next(iter(lst)) 


from pydoc import locate
import funcsigs
import inspect   
import requests
import shutil
import subprocess    
import tempfile
import unicodedata
import warnings


import os:  os.path.join; os.path.abspath; os.path.exists(some_path); os.system(‘执行shell命令’);  os.path.isdir(some_path);  os.makedirs;  os.stat;  os.remove(some_path);  
  grep " os\." */*/*.py | grep -v join | awk -F ":" '{print $2}' | grep os  





