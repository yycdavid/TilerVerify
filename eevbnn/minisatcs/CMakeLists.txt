#########################
# Important note: Minisat does not properly annotate its API with
# dllimport/dllexport, so the library build is forced to be static.
#########################
cmake_minimum_required(VERSION 3.5)
project(MiniSat VERSION 2.2 LANGUAGES CXX)

# Check if minisat is being used directly or via add_subdirectory, but allow overriding
if (NOT DEFINED MINISAT_MASTER_PROJECT)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(MINISAT_MASTER_PROJECT ON)
    else()
        set(MINISAT_MASTER_PROJECT OFF)
    endif()
endif ()

# options
option(MINISAT_BUILD_BINARIES "minisat: build binaries" ${MINISAT_MASTER_PROJECT})
option(MINISAT_BUILD_TESTING "Build and run MiniSat's tests" ON)
option(MINISAT_TEST_BENCHMARKS "Register benchmarks with CTest" OFF)
option(MINISAT_INSTALL "minisat: generate the install target" ON)
option(MINISAT_NOINLINE "disable inline (for easier profiling / debugging, etc)" OFF)
option(MINISAT_GPROF "enable gprof" OFF)
option(MINISAT_GPERF "enable gperftools for profiling" OFF)

find_package(ZLIB)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(libminisat STATIC
    # Impl files
    minisat/core/Solver.cc
    minisat/utils/Options.cc
    minisat/utils/System.cc
    minisat/simp/SimpSolver.cc
    # Header files for IDEs
    minisat/core/Dimacs.h
    minisat/core/Solver.h
    minisat/core/SolverTypes.h
    minisat/mtl/Alg.h
    minisat/mtl/Alloc.h
    minisat/mtl/Heap.h
    minisat/mtl/IntTypes.h
    minisat/mtl/Queue.h
    minisat/mtl/Sort.h
    minisat/mtl/Vec.h
    minisat/mtl/XAlloc.h
    minisat/utils/Options.h
    minisat/utils/ParseUtils.h
    minisat/utils/System.h
    minisat/utils/Random.h
    minisat/simp/SimpSolver.h
)

set_target_properties(libminisat
    PROPERTIES
        # Keep the library named as either libminisat.a or minisat.lib While
        # having the target's logical name be distinct from minisat (the
        # binary)
        OUTPUT_NAME "minisat"

        # build PIC binary so .a could be linked into a shared library
        POSITION_INDEPENDENT_CODE ON
)

## Add a namespace alias.
# This is useful to abstract over use of the library as installed vs subdirectory build
add_library(MiniSat::libminisat ALIAS libminisat)

target_link_libraries(libminisat ${ZLIB_LIBRARIES})

target_compile_features(libminisat
    PUBLIC
      cxx_attributes
      cxx_defaulted_functions
      cxx_deleted_functions
      cxx_final
)

target_include_directories(libminisat
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${ZLIB_INCLUDE_DIRS}
)

set(targets libminisat)
if (MINISAT_BUILD_BINARIES OR (MINISAT_BUILD_TESTING AND BUILD_TESTING) OR (MINISAT_TEST_BENCHMARKS AND BUILD_TESTING))
    # Also build two MiniSat executables
    # NOTE: `minisat` is used in tests
    add_executable(minisat
        minisat/core/Main.cc
    )
    target_link_libraries(minisat libminisat)


    add_executable(minisat-simp
        minisat/simp/Main.cc
    )
    target_link_libraries(minisat-simp libminisat)
    list(APPEND targets minisat minisat-simp)
endif()

# Workaround for libstdc++ + Clang + -std=gnu++11 bug.
set_target_properties(${targets}
    PROPERTIES
      CXX_EXTENSIONS OFF
)


foreach (target ${targets})
    if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU" )
        target_compile_options( ${target} PRIVATE -Wall -Wextra -Werror)
    endif()
    if ( CMAKE_CXX_COMPILER_ID MATCHES "MSVC" )
        target_compile_options( ${target} PRIVATE /W4 /wd4267 )
        target_compile_definitions( ${target} PRIVATE _CRT_SECURE_NO_WARNINGS )
    endif()
endforeach()

# fix compile options
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
if (MINISAT_NOINLINE OR MINISAT_GPROF)
    message("inline disabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fno-inline -fno-omit-frame-pointer")
endif()
if (MINISAT_GPROF)
    message("gprof enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -pg")
endif()
if (MINISAT_GPERF)
    message("gperftools enabled")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -lprofiler")
endif()

###############
# Testing
##
include(CTest)

if (MINISAT_BUILD_TESTING AND BUILD_TESTING)
    message(STATUS "Registering integration tests")
    # Read all easy instances from a file
    file(READ "${PROJECT_SOURCE_DIR}/tests/inputs/easy.txt" MINISAT_INTEGRATION_TESTS)
    string(REGEX REPLACE ";" "\\\\;" MINISAT_INTEGRATION_TESTS "${MINISAT_INTEGRATION_TESTS}")
    string(REGEX REPLACE "\n" ";" MINISAT_INTEGRATION_TESTS "${MINISAT_INTEGRATION_TESTS}")

    # Add 1 test for each easy instance
    foreach(INTEGRATION_TEST ${MINISAT_INTEGRATION_TESTS})
        add_test(NAME "integration:${INTEGRATION_TEST}"
            COMMAND minisat -verb=0 "tests/inputs/${INTEGRATION_TEST}"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        if ("${INTEGRATION_TEST}" MATCHES "^SAT")
            # The output can contain multiple lines, so we cannot do a full-line match
            # This means that we have to positive match SAT, but negative match UNSAT
            # because SAT is a substring of UNSAT.
            set_tests_properties("integration:${INTEGRATION_TEST}" PROPERTIES PASS_REGULAR_EXPRESSION "r=SAT\n"
                                                                              FAIL_REGULAR_EXPRESSION "r=UNSAT\n")
        else()
            set_tests_properties("integration:${INTEGRATION_TEST}" PROPERTIES PASS_REGULAR_EXPRESSION "r=UNSAT\n")
        endif()
        set_tests_properties("integration:${INTEGRATION_TEST}" PROPERTIES
            TIMEOUT 30
        ) # 30s timeout

        if ("${INTEGRATION_TEST}" MATCHES "(SAT|UNSAT)/ineq/.*")
            continue()
        endif()

        add_test(NAME "integration_simp:${INTEGRATION_TEST}"
            COMMAND minisat-simp -verb=0 "tests/inputs/${INTEGRATION_TEST}"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        if ("${INTEGRATION_TEST}" MATCHES "^SAT")
            # The output can contain multiple lines, so we cannot do a full-line match
            # This means that we have to positive match SAT, but negative match UNSAT
            # because SAT is a substring of UNSAT.
            set_tests_properties("integration_simp:${INTEGRATION_TEST}" PROPERTIES PASS_REGULAR_EXPRESSION "r=SAT\n"
                                                                                   FAIL_REGULAR_EXPRESSION "r=UNSAT\n")
        else()
            set_tests_properties("integration_simp:${INTEGRATION_TEST}" PROPERTIES PASS_REGULAR_EXPRESSION "r=UNSAT\n")
        endif()
        set_tests_properties("integration_simp:${INTEGRATION_TEST}" PROPERTIES
            TIMEOUT 30
        ) # 30s timeout
    endforeach(INTEGRATION_TEST)
endif() # TESTING


if (MINISAT_TEST_BENCHMARKS AND BUILD_TESTING)
    message(STATUS "Registering benchmarks")

    file(READ "${PROJECT_SOURCE_DIR}/tests/inputs/benchmarks.txt" MINISAT_BENCHMARKS)
    string(REGEX REPLACE ";" "\\\\;" MINISAT_BENCHMARKS "${MINISAT_BENCHMARKS}")
    string(REGEX REPLACE "\n" ";" MINISAT_BENCHMARKS "${MINISAT_BENCHMARKS}")

    # Add 1 test for each easy instance
    foreach(BENCHMARK ${MINISAT_BENCHMARKS})
        add_test(NAME "benchmark:${BENCHMARK}"
            COMMAND minisat -verb=0 "tests/inputs/${BENCHMARK}"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        if ("${BENCHMARK}" MATCHES "^SAT")
            # The output can contain multiple lines, so we cannot do a full-line match
            # This means that we have to positive match SAT, but negative match UNSAT
            # because SAT is a substring of UNSAT.
            set_tests_properties("benchmark:${BENCHMARK}" PROPERTIES PASS_REGULAR_EXPRESSION "r=SAT\n"
                                                                     FAIL_REGULAR_EXPRESSION "r=UNSAT\n")
        else()
            set_tests_properties("benchmark:${BENCHMARK}" PROPERTIES PASS_REGULAR_EXPRESSION "r=UNSAT\n")
        endif()
        set_tests_properties("benchmark:${BENCHMARK}" PROPERTIES
            TIMEOUT 86400
        ) # 1 day timeout
    endforeach(BENCHMARK)
endif()


###############
# Installation
##
if (MINISAT_INSTALL)
    set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/MiniSat)

    install(
        TARGETS
          libminisat
        EXPORT MiniSatTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    if (MINISAT_BUILD_BINARIES)
        install(
            TARGETS
              minisat
              minisat-simp
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    install(DIRECTORY minisat/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/minisat FILES_MATCHING PATTERN "*.h*")

    install(EXPORT MiniSatTargets
        FILE MiniSatTargets.cmake
        NAMESPACE MiniSat::
        DESTINATION ${INSTALL_CONFIGDIR}
    )

endif()
