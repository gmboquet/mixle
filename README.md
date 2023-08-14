pysparkplug is a Python package for distributed heterogeneous density estimation. With only a few lines of code you can specify and fit complex models on variable-length heterogenous data.


To install pysparkplug with make sure you have Python 3.7+ then run

    $ git clone https://github.com/LLNL/pysparkplug.git
    $ pip ./pysparkplug


Examples
----------------

Examples that run locally are located in ./pysp/examples/


    $ python pysp/examples/mixture_example.py


Examples that run with Apache Spark are located in./pysp/examples_spark/


    $ /path/to/spark/bin/spark-submit --master local[*] --py-files /path/to/package/dist/pysparkplug-0.1.8.4-py3.7.egg ./pysp/examples_spark/mixture_example.py 



License
----------------

pysparkplug is distributed under the terms of the MIT license.


LLNL-CODE-844837
