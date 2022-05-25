.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/PyVerletMD.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/PyVerletMD
    .. image:: https://readthedocs.org/projects/PyVerletMD/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://PyVerletMD.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/PyVerletMD/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/PyVerletMD
    .. image:: https://img.shields.io/pypi/v/PyVerletMD.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/PyVerletMD/
    .. image:: https://img.shields.io/conda/vn/conda-forge/PyVerletMD.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/PyVerletMD
    .. image:: https://pepy.tech/badge/PyVerletMD/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/PyVerletMD
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/PyVerletMD

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==========
PyVerletMD
==========


    A simple 2-dimensional molecular dynamics simulator using Verlet algorithm


<ADD VISILIZATION SAMPLE HERE>

Based on former McGill University MIME 473 midterm project.

So there was this project where we were asked to write a MATLAB script to perform a simple 2D molecular dynamics simulation implementing Verlet algorithm. 
Well I think one could just hard-code the script, but I made it kinda modular (I guess?) with many functions blahblah. 
I also add some sketchy live-plotting feature for better visualization (rather than writing a dump file and visulize in OVITO).
After I finished the first version of my MATLAB script, I gave it to some of my friends as a reference while I went on to optimize my code. 
It was a pretty interesting project, mainly because the live-plotting was kinda cool.
Anyway I now decide to re-write it in Python in a more "object-oriented" fashion.
The "not-so-object-oriented" version can be found in the legacy_code.py, where most things are hard-coded. 

Funny story:
2 years after I took MIME 473, I became the TA of this course. 
The midterm project did not change. 
When grading the codes submitted by students, I surprisingly found several scripts with hacky implementations that were way too familiar.
Apparently the my script got shared pretty widely. And after 2 years, no one cared to check who was the author of that poop mountain.
Another year later, professor had to change the topic of the midterm project (sorry prof.).

The moral of the story is: DON'T COPY-PASTE CODE without trying to UNDERSTAND it. 
I'll add more warning messages if this repo have more visitors.
<A comprehensive description which I have no time to finish>


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
