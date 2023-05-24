import argparse

def count_lines(input_file):
    with open(input_file) as file:
        num_lines = sum(1 for line in file)
        return num_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Number of lines')
    parser.add_argument('--input_file', type=str, help='file path')

    args = parser.parse_args()

    num_lines = count_lines(args.input_file)
    print(f'Number of lines: {num_lines}')