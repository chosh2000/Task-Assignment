class Vertex:
	def __init__(self, node):
		self.id = node
		self.current_processor = object
		self.k = {}


if __name__ == '__main__':

	# 3 types of vertices: node, processor, task
	tree_node = Vertex('n')
	processor1 = Vertex('p1')
	processor2 = Vertex('p2')
	task1 = Vertex('t1')
	task2 = Vertex('t2')
	tree_node.current_processor = processor1
	tree_node.k[task1] = processor1
	tree_node.k[task2] = processor2
	for t in tree_node.k:
		print("task %s is assigned to processor %s" % (t.id  ,  tree_node.k[t].id))					#prints task id and processor id in partial assignment k
		print(t ==task1)


	i = {4:1, 5:2, 6:3}
	j = {4:2, 7:7, 6:5}
	for k in j:
		if k in i:
			del i[k]
	print(i)

	# s = set()
	# s.add(task1)
	# s.add(task2)
	# print(1 in s)



	# dict = {'a','b'}
	# for keys in dict:
	# 	print 'a' in dict
	# print(dict[x] for x in dict)


	# print("dict['a'] = %d" %dict['a'])
	# print("dict['b'] = %d" %dict['b'])
	# if 'a' in dict:
	# 	print('yay')
	#  for key in dict:
	#  	if key in dict:
	# 		print('%s yay'%key)
	# a = min(20,10)
	# print('a=%d'%a)


	# for lel in dict:
	# 	print(lel)
	# 	print(dict[lel])
	# 	if (dict[lel] == 4):
	# 		print("lol = %d" %dict[lel])
