from sys import argv
from svm import svm


def read_args():
    i = 0
    args = {}
    while i < len(argv):
        if argv[i] == '-f':
            i += 1
            args['file'] = argv[i]

        elif argv[i] == '-c':
            i += 1
            start = (argv[i])
            i += 1
            end = int(argv[i])
            i += 1
            c = int(argv[i])
            args['cols'] = [start, end, c]

        elif argv[i] == '-r':
            i += 1
            start = argv[i]
            i += 1
            end = argv[i]
            args['train'] = [start, end]

        elif argv[i] == '-e':
            i += 1
            start = int(argv[i])
            i += 1
            end = int(argv[i])
            args['test'] = [start, end]

        elif argv[i] == '-l':
           i += 1
           start = int(argv[i])
           i += 1
           end = int(argv[i])
           args['classify'] = [start, end]

        elif argv[i] == '-t':
            i += 1
            args['rate'] = float(argv[i])
        elif argv[i] == '-i':
            i += 1
            args['epochs'] = int(argv[i])
        i += 1
    return args

def get_input(args):
    if 'file' not in args:
        file_path = input("----------------------\nEnter the file path for the csv: ")
    else:
        file_path = args['file']

    if 'cols' not in args:
        start_index = input("----------------------\nEnter the starting column index of the feature columns (0 starting index): ")
        end_index = input("----------------------\nEnter ending column index for feature columns(0 starting index): ")
        class_index = input("----------------------\nEnter the index of the classification column(0 starting index): ")
    else:
        start_index = args['cols'][0]
        end_index = args['cols'][1]
        class_index = args['cols'][2]

    if 'train' not in args:
        train_start_index = input("----------------------\nEnter the starting row index of the training data: ")
        train_end_index = input("----------------------\nEnter the ending index of the training data: ")
    else:
        train_start_index = args['train'][0]
        train_end_index = args['train'][1]

    if 'test' not in args:
        test_start_index = input("----------------------\nEnter the starting row index of the testing data: ")
        test_end_index = input("----------------------\nEnter the ending index of the testing data: ")
    else:
        test_start_index = args['test'][0]
        test_end_index = args['test'][0]

    try:
        if 'rate' not in args:
            rate = input('Enter learning rate (enter \"u\" if unkown) : ')
        else:
            rate = args['rate']

        if 'epochs' not in args:
            epochs = input('Enter number of iterations you would like the model to train (\"u\" if unknown): ')
        else:
            epochs = args['epochs']

        df = pd.read_csv(file_path)
        npa = np.asarray(df)
        train_X = npa[int(train_start_index):int(train_end_index)][int(start_index):int(end_index)+1]
        train_y = npa[int(train_start_index):int(train_end_index)][int(class_index)]
        test_X = npa[int(test_start_index):int(test_end_index)][int(start_index):int(end_index)+1]
        test_y = npa[int(test_start_index):int(test_end_index)][int(class_index)]

    except:
        print("\n\nOops! something was wrong with your input. Please try again...")
        return get_input()

    return [train_X, train_y, test_X, test_y, float(rate), int(epochs)]

args = read_args()
data_segments = get_input(args)

w = svm(data_segments[0], data_segments[1], data_segments[4], data_segments[5])

errors, tot = 0, 0
for x, y in zip(data_segments[2], data_segments[3]):
    if y * np.dot(x, w) < 1:
        errors += 1
    tot += 1

print("testing error percentage: %", 100 * (errors / tot),)
