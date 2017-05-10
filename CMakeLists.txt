cmake_minimum_required(VERSION 3.6)
project(C_Voxels)

set(CMAKE_CXX_STANDARD 11)

set(Python_ADDITIONAL_VERSIONS 2.7)



list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)


find_package(numpy)
find_package(PythonLibs REQUIRED)

if (PYTHONLIBS_FOUND)
	MESSAGE(STATUS "Found Python")
	MESSAGE(STATUS "PYTHON_INCLUDE_DIRS =  ${PYTHON_INCLUDE_DIRS}")
	MESSAGE(STATUS "PYTHON_LIBRARIES    =  ${PYTHON_LIBRARIES}")
else (PythonLibs_FOUND)
	MESSAGE(FATAL_ERROR "Couldn't find Python")
endif(PYTHONLIBS_FOUND)

set(SOURCE_FILES
        c_voxels.c
        c_voxels.h
		conversion_utilities.h
		conversion_utilities.c
        uthash.h
        utlist.h)

add_library(C_Voxels SHARED ${SOURCE_FILES})


include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES})
target_link_libraries(C_Voxels ${PYTHON_LIBRARIES})




if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()



# windows:
#cmake -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# make sure the "python" commands calls the right python version interp


