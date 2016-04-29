# **GLIA**: **G**raph **L**earning Library for **I**mage **A**nalysis #

### What? ###

A C++11 library for efficient hierarchical image segmentation. 

Please cite the following papers accordingly if you use the code: 

* T. Liu, C. Jones, M. Seyedhosseini, T. Tasdizen. A modular hierarchical approach to 3D electron microscopy image segmentation. Journal of Neuroscience Methods, 226, pp. 88--102, 2014.

* T. Liu, E. Jurrus, M. Seyedhosseini, T. Tasdizen. Watershed merge tree classification for electron microscopy image segmentation. ICPR 2012.

* T. Liu, M. Seyedhosseini, T. Tasdizen. Image segmentation using hierarchical merge tree. arXiv: 1505.06389v2, 2015.

### How? ###

Use a modern compiler with C++11 support, e.g., GCC-4.8 or higher and Apple LLVM 6.

Dependencies:

* InsightToolkit (ITK).
* Boost C++ libraries.
* Eigen.

Instructions:

* Use '-DCMAKE_CXX_FLAGS=-std=c++11' for the first time ITK CMake configuration.
* Turn on 'ITKReview' module for ITK.
* Enable C++11 for Boost libraries.

CMake configurations:

* Turn on 'GLIA_MT' to use OpenMP parallelization.
* Work on 3D/2D images with 'GLIA_3D' turned on/off.
* Turn on 'GLIA_BUILD_{HMT,GADGET,ML_RF}' modules accordingly.
* The random forest classifier used in our code is based on work of Andy Liaw (https://cran.r-project.org/web/packages/randomForest/index.html). To use the related functionalities, please set 'RF_SRC_DIR' as the path to 'randomForest/src/' folder in their code.

### Who? ###

* Ting Liu <<tingianliu@gmail.com>>
