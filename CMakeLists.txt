cmake_minimum_required(VERSION 3.6)
project(C_Voxels)

set(CMAKE_CXX_STANDARD 11)

set(Python_ADDITIONAL_VERSIONS 2.7)



list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)


find_package(numpy)

set(SOURCE_FILES
        c_voxels.c
        c_voxels.h
		conversion_utilities.h
		conversion_utilities.c
        uthash.h
        utlist.h)

add_library(C_Voxels SHARED ${SOURCE_FILES})

# Woaw ugly shit
if (UNIX)
    if (CYGWIN)
        set(PYTHON_INCLUDE_DIR C:\\Python27\\include)
        set(PYTHON_LIBRARY C:\\Python27\\libs)
        include_directories(${PYTHON_INCLUDE_DIR})
        link_libraries(C_Voxels, ${PYTHON_LIBRARY})
    ENDIF()
    find_package(PythonLibs)
    IF(PYTHONLIBS_FOUND)
        include_directories(${PYTHON_INCLUDE_DIRS})
        link_libraries(C_Voxels, ${PYTHON_LIBRARIES})
    ELSEIF(NOT PYTHONLIBS_FOUND)
        MESSAGE(FATAL_ERROR "Unable to find correct Python version. Did you set PythonLibs?")
    ENDIF(PYTHONLIBS_FOUND)
ENDIF()

if(WIN32)
	#set(CMAKE_SUPPRESS_REGENERATION true)
    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES})
    target_link_libraries(C_Voxels ${PYTHON_LIBRARIES})

	MESSAGE(STATUS ${PYTHON_INCLUDE_DIRS})
	MESSAGE(STATUS ${PYTHON_LIBRARIES})


    #set(PYTHON_INCLUDE_DIR C:\\Python27\\include)
    #set(PYTHON_LIBRARY C:\\Python27\\libs)
    #include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDES})
    #link_libraries(C_Voxels ${PYTHON_LIBRARY} ${NUMPY_LIBRARIES})
ENDIF()



# windows:
#cmake -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# make sure the "python" commands calls the right python version interp


