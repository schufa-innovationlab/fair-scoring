How To Set Sphinx Up
====================

Generating HTML Docs
--------------------
Sphinx is already set up to generate html output.
Just call the following two commands from the ``./docs``

.. sourcecode:: shell

   make clean
   make html


Resolving the TODOs
-------------------
There are some TODOs left here to be resolved before generating the first "real" docs.

These are currently marked as warnings like the following one:

.. warning::
   **TODO**: Remove this file and the example notebook and create real examples and tutorials.

If the package has no examples / tutorials, just remove:

- the corresponding folder
- the ``examples.rst`` / ``tutorials.rst``
- the toc-entry in the ``index.rst``

Extending the documentation
---------------------------
- You can use restucturedText (``.rst``), markdown (``.md``) and jupyter notebooks (``.ipynb``) for the documentation.
- Tutorials and Examples are automatically included from the corresponding subfolders
  (Have a look at ``examples.rst`` / ``tutorials.rst`` if you want to add another category)
- Top-Level files can be referenced in the ``index.rst``