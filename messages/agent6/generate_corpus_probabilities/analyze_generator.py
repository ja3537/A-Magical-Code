import os

if __name__ == "__main__":

	agent_list = {}

	os.chdir(os.path.dirname(os.path.realpath(__file__)))
	agents = [1,2,3,4,5,6,7,8]

	for agent_number in agents:

		count = {}
		agent_name = "./agent{}".format(agent_number)
		print(agent_name)

		for filename in os.listdir(agent_name):
			print(agent_name + "/" + filename)

			f = open(agent_name + "/" + filename, "r")
			
			for line in f:

				#clean line
				line = line.strip()

				#count characters
				for c in line:
					count[c] = count[c]+1 if c in count else 1
		
			f.close()
		
		agent_list[agent_number] = (count)

	print(agent_list)