import sys
from setuptools import find_packages, setup

if sys.version_info < (2, 7, 0) or sys.version_info >= (3, 0, 0):
   print "Python 2.7 required"

try:
    import cv, cv2
    from cv2 import __version__
    if __version__ < '2.4.7':
        print "Please update OpenCV to version >=2.4.7"
        exit()
except ImportError:
    print "Please install OpenCV with Python bindings in order to continue."
    exit()

setup(name="audetect",
      version="0.1",
      description="Action Unit ",
      author="Derek Schultz",
      author_email='derek@derekschultz.net',
      platforms=["any"],
      license="BSD",
      url="http://github.com/derek-schultz/audetect",
      packages=find_packages(),
      install_requires=["numpy>=1.6"],
      scripts=["audetect/audetect-interactive.py"],
      )
