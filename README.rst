BOOK RECOMMENDER
================
This is school project for Recommender Systems course on Masaryk University, Faculty of Informatics. The goal of this
project is to design, implement and evaluate Recommender system for online book store.

Dataset
-------

Dataset analysis:
https://github.com/matejkvassay/book-recommender/blob/master/ipython_notebooks/data_analysis.ipynb

Download CSV dump:
http://www2.informatik.uni-freiburg.de/~cziegler/BX/

After download set paths to csv files in config.yml.


DEVELOPMENT
===========

Install requirements manually if needed:

.. code-block:: shell
    pip install -r requirements.txt

Install package for developement:

.. code-block:: shell

    python setup.py develop

Install package for production:

.. code-block:: shell

    python setup.py install

Run tests:

.. code-block:: shell

    py.test

Commit:

.. code-block:: shell

    git pull
    git commit -am 'commit message'
    git push

IPYTHON NOTEBOOKS
=================

http://jupyter.org

Run notebooks in browser:

.. code-block:: shell

    jupyter notebook



