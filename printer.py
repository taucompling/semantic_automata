import logging


def set_up_logging(output_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh = logging.FileHandler(output_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def info(*args):
    logging.getLogger().info(' '.join(map(str, args)))


if __name__ == '__main__':
    set_up_logging('tmp.log')
    info('This is a message')
    info('This is another message: %d, %.2f' % (7, 7.7), 'Yay')