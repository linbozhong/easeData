# coding:utf-8

import os
import logging
from const import *
from functions import getDataDirectory
from datetime import datetime


class Logger(object):
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.level = None

        self.consoleHandler = None
        self.fileHandler = None

        # self.setLevel(logging.INFO)
        # self.addConsoleHandler()
        # self.addFileHandler()

    def setLevel(self, level):
        self.logger.setLevel(level)
        self.level = level

    def addConsoleHandler(self, level=None):
        if not self.consoleHandler:
            if level is None:
                level = self.level
            self.consoleHandler = logging.StreamHandler()
            self.consoleHandler.setLevel(level)
            self.consoleHandler.setFormatter(self.formatter)
            self.logger.addHandler(self.consoleHandler)

    def addFileHandler(self, level=None, filename=None):
        if not self.fileHandler:
            if level is None:
                level = self.level
            if filename is None:
                filename = '{}.log'.format(datetime.now().strftime('%Y-%m-%d'))
            folder = os.path.join(getDataDirectory(), DIR_LOG)
            if not os.path.exists(folder):
                os.makedirs(folder)
            fp = os.path.join(folder, filename)
            self.fileHandler = logging.FileHandler(fp)
            self.fileHandler.setLevel(level)
            self.fileHandler.setFormatter(self.formatter)
            self.logger.addHandler(self.fileHandler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)