class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'C:\\Users\\zlee020\\Desktop\\UCF101'

            # Save preprocess data into output_dir
            output_dir = 'C:\\Users\\zlee020\\Desktop\\outputpic'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'C:\\Users\\zlee020\\Desktop\\c3d-pretrained.pth'