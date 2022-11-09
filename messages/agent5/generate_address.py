import random


def generate_address(n, outfile):

    write_file = open(outfile, "w")

    street_suff_file = open('messages/agent5/street_suffix.txt')
    street_suff = street_suff_file.readlines()
    suff_num = len(street_suff)

    street_name_file = open('messages/agent5/street_name.txt')
    street_name = street_name_file.readlines()
    name_num = len(street_name)

    #address_list = []

    for address in range(n):

        current_address = []

        num_length = random.randint(1, 4)

        digit_list = []

        for i in range(num_length):
            digit = random.randint(0, 9)
            digit_list.append(str(digit))

        digit_string = ''.join(digit_list)
        current_address.append(digit_string)

        street_num = random.randint(1, name_num)
        name = street_name[street_num]
        name = name[:-1]

        current_address.append(name)

        suffix_num = random.randint(1, suff_num)
        suffix = street_suff[suffix_num]
        suffix = suffix[:-1]

        current_address.append(suffix)

        if address < n - 1:
            current_address.append("\n")

        current_address_string = ' '.join(current_address)

        write_file.write(current_address_string)


    return True


if __name__ == "__main__":

    addresses = generate_address(25, "messages/addresses_out.txt")

    #print(addresses)


