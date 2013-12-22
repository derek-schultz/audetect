audetect
========

Recognizes facial expressions using computer vision. Given a series of input images (video), returns a list of active [facial action units](http://www.cs.cmu.edu/~face/facs.htm).

Methods are based loosely on [Valstar & Pantic 2006](http://ibug.doc.ic.ac.uk/media/uploads/documents/CVPR06-ValstarPantic-FINAL.pdf).

A (boring) [video demonstration](http://www.youtube.com/watch?v=l1m_9wQ-Dnk) of the program (slowly) doing its thing.

Requirements
------------
audetect has only been tested with

* Python 2.7
* OpenCV 2.4.7

Installation
------------
```
# python setup.py install
```

Running
-------
```
$ audetect-interactive.py --help
```
should give you a good start.
