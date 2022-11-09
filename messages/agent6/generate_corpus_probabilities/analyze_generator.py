import os

if __name__ == "__main__":

	agents = [1,2,3,4,5,6,7,8]

	for agent_number in agents:

		count = {}
		agent_name = "agent{}".format(agent_number)

		for filename in os.listdir(agent_name):

			f = open(filename, "r")
			
			for line in f:

				#clean line
				line = line.strip()

				#count characters
				for c in line:
					count[c] = count[c]+1 if c in count else 1
		
			f.close()