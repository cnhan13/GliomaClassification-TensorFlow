from os import walk

def gen(dir_path):
  file_L = []
  file_H = []
  file_ = []
  
  for (_, _, filenames) in walk(dir_path):
    file_.extend(filenames)
    break

  
  
def main(argv=None):
  gen(dir_path)

if __name__ == '__main__':
  tf.app.run()
