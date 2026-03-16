.. YRT-PET documentation master file, created by
   sphinx-quickstart on Tue Mar 11 15:45:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

YRT-PET documentation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Compilation

   compilation/building
   compilation/adding_plugins

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/index
   usage/config

.. toctree::
   :maxdepth: 2
   :caption: File formats

   usage/data_formats
   usage/scanner
   usage/image_parameters
   usage/list-mode_file
   usage/motion_file
   usage/histogram3d_format
   usage/rawd_file
   usage/sparse-histogram
   usage/ImagePSF_file

.. toctree::
   :maxdepth: 2
   :caption: Python

   python/owned_vs_alias

.. toctree::
   :maxdepth: 2
   :caption: Other

   faq
   compilation/contributing

Citing YRT-PET:
---------------
If you use YRT-PET in one of your projects, you can cite our
`journal article`_ in *IEEE Transactions on Radiation and Plasma Medical
Sciences*.

.. _`journal article`: https://ieeexplore.ieee.org/document/11202639

.. code-block:: LaTeX

    @article{najmaoui2025,
        title = {{YRT}-{PET}: An Open-Source {GPU}-Accelerated Image
            Reconstruction Engine for Positron Emission Tomography},
        issn = {2469-7303},
        url = {https://ieeexplore.ieee.org/document/11202639},
        doi = {10.1109/TRPMS.2025.3619872},
        journal = {{IEEE} Transactions on Radiation and Plasma Medical
            Sciences},
        pages = {1--1},
        author = {Najmaoui, Yassir and Chemli, Yanis and Toussaint, Maxime and
            Petibon, Yoann and Marty, Baptiste and Fontaine, Kathryn and
            Gallezot, Jean-Dominique and Razdevšek, Gašper and Orehar, Matic and
            Dhaynaut, Maeva and Guehl, Nicolas and Dolenec, Rok and
            Pestotnik, Rok and Johnson, Keith and Ouyang, Jinsong and
            Normandin, Marc and Tétrault, Marc-André and Lecomte, Roger and
            El Fakhri, Georges and Marin, Thibault},
        year = {2025},
        keywords={Image reconstruction;Biomedical imaging;Graphics processing
            units;Software;Positron emission tomography;Python;Engines;Software
            algorithms;Plasmas;Geometry;positron emission tomography (PET);image
            reconstruction;software;graphics processing unit (GPU);C++;Python}
    }
