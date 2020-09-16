#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from posenetGestures import posenetGesture

if __name__ == '__main__':
    app = posenetGesture()
    app.load_model()
    app.start()
