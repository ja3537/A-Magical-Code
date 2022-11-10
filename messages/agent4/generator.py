import random

def generate(numMessages, outfile):
    output = open(outfile, "w")

    v_directions = ["N", "S"]
    h_directions = ["E", "W"]

    for i in range(numMessages):
        latitude = format_num(90)
        longitude = format_num(180)
        random_dir_lat = random.choice(v_directions)
        random_dir_long = random.choice(h_directions)
        location = latitude + " " + random_dir_lat + ", " + longitude + " " + random_dir_long + "\n"
        output.write(location)

def format_num(max_range):
    num = round(random.uniform(0, max_range), 4)
    num_str = str(num)
    int_part = num_str.split('.')[0]
    dec_part = num_str.split('.')[1]
    dec_with_zeros = dec_part.ljust(4, '0')
    return "" + int_part + "." + dec_with_zeros

if __name__ == "__main__":
    generate(20, "locations_out.txt")