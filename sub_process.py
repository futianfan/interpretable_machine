from subprocess import Popen, PIPE

cmd = ['ls', '-l']

subproc = Popen(cmd, stdin = PIPE, stdout = PIPE)


#Input = subproc.stdin
#print('1', file = subproc.stdin)


output = subproc.stdout.readlines()


for elem in output:
	print(elem)

