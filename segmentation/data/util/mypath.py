#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os


class Path(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/home/vrai/dataset' # VOC will be automatically downloaded
        #db_root = '/home/david/Documents/repos/progetto-cv/Unsupervised-Semantic-Segmentation/pretrain/PASCAL_VOC' # VOC will be automatically downloaded

        
        db_names = ['VOCSegmentation','imaterialist-fashion-2020-fgvc7']

        if database == '':
            return db_root

        if database in db_names:
            return os.path.join(db_root, database)

        else:
            raise ValueError('Invalid database {}'.format(database))
