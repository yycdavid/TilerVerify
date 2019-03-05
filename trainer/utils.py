import os
import sys

class OutputManager(object):
    def __init__(self, result_path):
        self.log_file = open(os.path.join(result_path,'log.txt'),'w')

    def say(self, s):
        self.log_file.write("{}".format(s))
        self.log_file.flush()
        sys.stdout.write("{}".format(s))
        sys.stdout.flush()

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)
    else:
        raise Exception('Result folder for this experiment already exists')
