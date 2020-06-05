#
# Classes not derived from New-Style classes are Old-Style classes.
# The built-in super() does not work in them:
#

class baseClass1:
	def __init__(self):
		self.a = 1
		self.b = 'Privet!'
		print 'Constructor of baseClass1'

	def fun1(self):
		print 'Fun1() called'
	def fun2(self):
		print 'Fun2() called'
		
		
class deriv1(baseClass1):
	def __init__(self):
		#super(deriv1,self).__init__()
		#
		# An alternative to super() in New-Style classes:
		#
		baseClass1.__init__(self)
		self.c = 3
		self.d = 'Hello!'
		print 'Constructor of deriv'
		
	def fun3(self):
		print 'Fun3() called'
	def fun4(self):
		print 'Fun4() called'
	

#
# Classes derived from the most base type 'object' are New-Style classes
# or from New-Style classes are New-Style classes.
# The built-in super() works in them:
#

class P(object):
	def __init__(self):
		self.a = 1
		self.b = 'Privet!'
		print 'Constructor of P'

		
class Q(P):
	def __init__(self):
		#super().__init__()
		super(Q,self).__init__()
		#P()
		self.c = 3
		self.d = 'Hello!'
		print 'Constructor of Q'





